"""Frame cache (decoded once), token cache, readout-site sweep, probe-set builder; gates G1, G2, G5.

Layout under cfg['cache_dir']:
    frames/<suite>.npy            uint8 (N,256,256,3), dataset orientation, rows in dataset 'index' order
    frames/<suite>.npz            ep, t, task (LIBERO task_id), state (N,8), action (N,7), tasks (10 strings)
    tokens/<model>/<site>__<text>/<suite>.npy   float16 (N,64,D)  (+ .stats.npz: per-dim mean/std)
    probe/frames.npy, probe/labels.npz          fixed probe set shared by every model
    probe/init_<suite>.npz                      first env frame/state for each of the 50 init states
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from repstudy import policy
from repstudy.encoders import GENERIC_PROMPT, load
from repstudy.run import log, site_for

# ---------------------------------------------------------------- frame cache


def _frames_dir(cfg):
    return Path(cfg["cache_dir"]) / "frames"


def _decode_episode(args):
    path, ts = args
    from lerobot.datasets.video_utils import decode_video_frames
    x = decode_video_frames(path, ts, tolerance_s=1e-4, backend="torchcodec", return_uint8=True)
    return x.permute(0, 2, 3, 1).contiguous().numpy()


def snapshot(cfg):
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(cfg["dataset"], repo_type="dataset", revision=cfg["revision"],
                                  allow_patterns=["meta/*", "data/*", "videos/observation.images.image/*"]))


def read_dataset(root):
    """Per-frame table + episode table + {task_index: instruction}, straight from parquet."""
    import pandas as pd
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    task_str = {int(i): s for s, i in zip(tasks.index, tasks["task_index"])}
    eps = pd.concat([pd.read_parquet(f) for f in sorted((root / "meta" / "episodes").rglob("*.parquet"))])
    cols = ["index", "episode_index", "frame_index", "timestamp", "task_index", "observation.state", "action"]
    data = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted((root / "data").rglob("*.parquet"))])
    return data.sort_values("index").reset_index(drop=True), eps.set_index("episode_index"), task_str


def build_frame_cache(cfg):
    import multiprocessing as mp
    root = snapshot(cfg)
    info = json.loads((root / "meta" / "info.json").read_text())
    data, eps, task_str = read_dataset(root)
    out = _frames_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    key = "observation.images.image"
    for suite in cfg["suites"]:
        if (out / f"{suite}.npz").exists():
            continue
        names = policy.task_names(suite)
        tid = {i: names.index(s) for i, s in task_str.items() if s in names}
        assert len(tid) == len(names), f"{suite}: {len(tid)}/{len(names)} LIBERO task strings found"
        d = data[data["task_index"].isin(list(tid))]
        ep_ids = d["episode_index"].unique()
        jobs = []
        for e in ep_ids:
            r = eps.loc[e]
            path = root / info["video_path"].format(video_key=key, chunk_index=int(r[f"videos/{key}/chunk_index"]),
                                                    file_index=int(r[f"videos/{key}/file_index"]))
            de = d[d["episode_index"] == e]
            assert (de["frame_index"].to_numpy() == np.arange(len(de))).all(), f"episode {e} not contiguous"
            jobs.append((str(path), (r[f"videos/{key}/from_timestamp"] + de["timestamp"].to_numpy(np.float64)).tolist()))
        frames = np.lib.format.open_memmap(out / f"{suite}.npy.tmp", mode="w+", dtype=np.uint8,
                                           shape=(len(d), cfg["image_size"], cfg["image_size"], 3))
        i = 0
        with mp.get_context("forkserver").Pool(cfg["n_workers"]) as pool:   # parent may hold CUDA
            for x in pool.imap(_decode_episode, jobs, chunksize=4):
                frames[i:i + len(x)] = x
                i += len(x)
        assert i == len(d), (i, len(d))
        frames.flush()
        del frames
        (out / f"{suite}.npy.tmp").replace(out / f"{suite}.npy")
        np.savez(out / f"{suite}.npz", ep=d["episode_index"].to_numpy(), t=d["frame_index"].to_numpy(),
                 task=d["task_index"].map(tid).to_numpy(), state=np.stack(d["observation.state"]).astype(np.float32),
                 action=np.stack(d["action"]).astype(np.float32), tasks=np.array(names))
        log(f"frame cache {suite}: {len(d)} frames, {len(ep_ids)} episodes")


def load_frames(cfg, suite):
    p = _frames_dir(cfg) / f"{suite}.npz"
    if not p.exists():
        build_frame_cache(cfg)
    m = dict(np.load(p))
    m["frames"] = np.load(_frames_dir(cfg) / f"{suite}.npy", mmap_mode="r")
    m["tasks"] = [str(s) for s in m["tasks"]]
    return m


# ---------------------------------------------------------------- token cache


def texts_for(fc, rows, text):
    if text == "generic":
        return [GENERIC_PROMPT] * len(rows)
    return [fc["tasks"][k] for k in fc["task"][rows]]


def encode_rows(enc, fc, rows, sites, text, batch):
    """-> {site: float16 (len(rows),64,D)} kept in RAM."""
    out = {}
    for a in range(0, len(rows), batch):
        r = rows[a:a + batch]
        z = enc.encode(np.asarray(fc["frames"][r]), texts_for(fc, r, text), sites)
        for s, v in z.items():
            if s not in out:
                out[s] = np.empty((len(rows),) + tuple(v.shape[1:]), np.float16)
            out[s][a:a + len(r)] = v.numpy()
    return out


def token_path(cfg, model, site, text, suite):
    return Path(cfg["cache_dir"]) / "tokens" / model / f"{site.replace('/', '_')}__{text}" / f"{suite}.npy"


def token_stats(x, chunk=4096):
    """Per-dimension mean/std over all frames and token positions (float64 accumulation)."""
    s = s2 = 0.0
    n = 0
    for a in range(0, len(x), chunk):
        v = np.asarray(x[a:a + chunk], np.float64).reshape(-1, x.shape[-1])
        s, s2, n = s + v.sum(0), s2 + (v * v).sum(0), n + len(v)
    mu = s / n
    return mu.astype(np.float32), np.sqrt(np.maximum(s2 / n - mu * mu, 1e-12)).astype(np.float32)


def extract(cfg, model, res, device, text):
    if model == "blind":
        return log("blind: no tokens")
    if policy.cut_closed_loop(cfg, model, text):
        return log(f"{model}/{text}: closed loop cut (cut_level {cfg['cut_level']}): no tokens")
    enc = load(model, device)
    site = site_for(cfg, model, enc.sites)
    for suite in cfg["suites"]:
        p = token_path(cfg, model, site, text, suite)
        if policy.heads_done(cfg, model, site, suite, text):   # train deletes tokens once heads exist
            continue
        if p.exists():
            log(f"tokens cached: {p}")
            continue
        fc = load_frames(cfg, suite)
        n, B = len(fc["ep"]), cfg["encode_batch"]
        p.parent.mkdir(parents=True, exist_ok=True)
        mm, t0 = None, time.time()
        for a in range(0, n, B):
            r = np.arange(a, min(a + B, n))
            z = enc.encode(np.asarray(fc["frames"][r]), texts_for(fc, r, text), [site])[site]
            if mm is None:
                mm = np.lib.format.open_memmap(str(p) + ".tmp", mode="w+", dtype=np.float16, shape=(n,) + tuple(z.shape[1:]))
            mm[r] = z.numpy()
        mm.flush()
        mu, sd = token_stats(mm)
        del mm
        np.savez(p.with_suffix(".stats.npz"), mean=mu, std=sd)   # stats first: .npy present => stats present
        Path(str(p) + ".tmp").replace(p)
        log(f"{model}/{site}/{suite}: {n} frames, {n / (time.time() - t0):.0f} frames/s")
    res.log("site", "extracted", site)


# ---------------------------------------------------------------- site sweep (§1.4)


def calibrate(cfg, model, res, device, text):
    """Brief head training on ~10% of episodes per suite at every candidate site; held-out chunk L1."""
    if model == "blind":
        return log("blind: no sites")
    if text == "instruction" and model in cfg["prompt_invariant"]:
        res.log("calibrate", text, {"same_as": "generic", "why": "prompt-invariant image tokens (G5)"})
        return log(f"{model}: instruction-on equals generic by construction")
    if res.get("calibrate", text) and res.get("calibrate", f"{text}_suites") == cfg["suites"]:
        return log(f"{model}/{text}: calibrated")   # never re-pick a site after rollouts exist
    enc = load(model, device)
    c = cfg["calib"]
    curve = {s: [] for s in enc.sites}
    for suite in cfg["suites"]:
        fc = load_frames(cfg, suite)
        rng = np.random.default_rng(0)
        eps = np.unique(fc["ep"])
        pick = rng.choice(eps, max(5, int(round(c["frac"] * len(eps)))), replace=False)
        held = set(pick[: max(1, int(round(c["heldout"] * len(pick))))].tolist())
        rows = np.flatnonzero(np.isin(fc["ep"], pick))
        tr = np.array([fc["ep"][r] not in held for r in rows])
        toks = encode_rows(enc, fc, rows, enc.sites, text, cfg["encode_batch"])
        for site, x in toks.items():
            mu, sd = token_stats(x[tr])
            data = policy.make_data(cfg, x, mu, sd, fc, rows, suite, device)
            head = policy.fit(cfg, data, np.flatnonzero(tr), seed=0, steps=c["steps"], device=device)
            curve[site].append(policy.chunk_l1(head, data, np.flatnonzero(~tr)))
        log(f"{model}/{suite}: " + ", ".join(f"{s}={v[-1]:.4f}" for s, v in curve.items()))
    curve = {s: float(np.mean(v)) for s, v in curve.items()}
    res.log("calibrate", text, curve)
    res.log("calibrate", f"{text}_suites", cfg["suites"])
    if text == "generic":
        site = site_for(cfg, model, enc.sites)
        res.log("site", "name", site)
        res.log("site", "index", enc.sites.index(site))
        res.log("site", "rule", cfg["site_rule"].get(model, "argmin"))
        log(f"{model}: site {site} (curve {curve})")


# ---------------------------------------------------------------- probe set (§1.6)


def init_frames(cfg, suite):
    """First env observation (dataset orientation) + 8-d state for every init state of every task."""
    p = Path(cfg["cache_dir"]) / "probe" / f"init_{suite}.npz"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        fr, st = [], []
        for tid in range(len(policy.task_names(suite))):
            f, s = policy.reset_frames(suite, tid, range(policy.n_init_states(suite, tid)))
            fr.append(f)
            st.append(s)
        np.savez(p, frames=np.stack(fr), state=np.stack(st))
    return dict(np.load(p))


def match_init(fc, init, ep):
    """Nearest init state for a demo episode by first-frame pixel MSE. -> (index, mse, mse_unrotated, state_err)."""
    r = np.flatnonzero(fc["ep"] == ep)[0]
    f0 = fc["frames"][r].astype(np.float32)
    cand = init["frames"][fc["task"][r]].astype(np.float32)          # (50,H,W,3)
    mse = ((cand - f0) ** 2).mean((1, 2, 3))
    k = int(mse.argmin())
    unrot = float(((cand[k, ::-1, ::-1] - f0) ** 2).mean())
    return k, float(mse[k]), unrot, float(np.abs(init["state"][fc["task"][r], k] - fc["state"][r]).max())


def build_probe_set(cfg):
    """~6k dataset frames (robot state, progress, task) + ~8k replayed-demo frames that also carry the
    target-object position and gripper->object offset. Shared by every model."""
    d = Path(cfg["cache_dir"]) / "probe"
    if (d / "labels.npz").exists():
        return
    d.mkdir(parents=True, exist_ok=True)
    P, rng = cfg["probe"], np.random.default_rng(0)
    F, L = [], {k: [] for k in ("source", "task", "group", "progress", "state", "obj", "offset")}

    def add(frames, task, group, progress, state, obj, src):
        F.append(frames)
        n = len(frames)
        for k, v in (("source", np.full(n, src)), ("task", np.full(n, task)), ("group", np.broadcast_to(group, n)),
                     ("progress", progress), ("state", state), ("obj", obj), ("offset", obj - state[:, :3])):
            L[k].append(np.asarray(v, np.float32 if k in ("progress", "state", "obj", "offset") else np.int64))

    fcs = {s: load_frames(cfg, s) for s in cfg["suites"]}
    per_suite = P["n_dataset"] // len(cfg["suites"])
    for si, (suite, fc) in enumerate(fcs.items()):
        ep_ids, start, length = np.unique(fc["ep"], return_index=True, return_counts=True)
        r = np.sort(rng.choice(len(fc["ep"]), per_suite, replace=False))
        n_ep = length[np.searchsorted(ep_ids, fc["ep"][r])]
        for task in np.unique(fc["task"][r]):   # add() takes one task id per call
            m = fc["task"][r] == task
            rr = r[m]
            add(np.asarray(fc["frames"][rr]), si * 10 + task, fc["ep"][rr], fc["t"][rr] / np.maximum(1, n_ep[m] - 1),
                fc["state"][rr], np.full((len(rr), 3), np.nan, np.float32), 0)
    # replay: round-robin over (suite, task), one demo episode at a time, until n_replay frames
    inits = {s: init_frames(cfg, s) for s in cfg["suites"]}
    queues = [(si, suite, tid, list(rng.permutation(np.unique(fc["ep"][fc["task"] == tid]))))
              for si, (suite, fc) in enumerate(fcs.items()) for tid in range(10)]
    got = 0
    while got < P["n_replay"] and any(q[3] for q in queues):
        for si, suite, tid, eps in queues:
            if not eps or got >= P["n_replay"]:
                continue
            e = eps.pop()
            fc = fcs[suite]
            k = match_init(fc, inits[suite], e)[0]
            fr, st, obj, _ = policy.replay(suite, tid, k, fc["action"][fc["ep"] == e])
            keep = np.arange(0, len(fr), P["replay_stride"])
            # group = dataset episode id for both sources, so CV folds never split a demo from its replay
            add(fr[keep], si * 10 + tid, e, keep / max(1, len(fr) - 1), st[keep], obj[keep], 1)
            got += len(keep)
    frames = np.concatenate(F)
    np.save(d / "frames.npy", frames)
    np.savez(d / "labels.npz", **{k: np.concatenate(v) for k, v in L.items()})
    log(f"probe set: {len(frames)} frames ({got} replayed)")


def load_probe_set(cfg):
    d = Path(cfg["cache_dir"]) / "probe"
    build_probe_set(cfg)
    return np.load(d / "frames.npy", mmap_mode="r"), dict(np.load(d / "labels.npz"))


# ---------------------------------------------------------------- gates


def gate_g1(cfg):
    """Dataset contract: keys/shapes, suite boundaries, 30/30 task strings, pinned revision."""
    root = snapshot(cfg)
    assert cfg["revision"] in str(root), f"snapshot {root} is not revision {cfg['revision']}"
    info = json.loads((root / "meta" / "info.json").read_text())
    f = info["features"]
    assert info["codebase_version"] == "v3.0", info["codebase_version"]
    assert f["observation.images.image"]["shape"] == [256, 256, 3], f["observation.images.image"]
    assert f["observation.state"]["shape"] == [8] and f["action"]["shape"] == [7]
    data, eps, task_str = read_dataset(root)
    report = {}
    for suite in cfg["suites"]:
        names = policy.task_names(suite)
        found = [s for s in names if s in task_str.values()]
        ids = [i for i, s in task_str.items() if s in names]
        e = np.unique(data[data["task_index"].isin(ids)]["episode_index"])
        assert len(found) == len(names) == 10, (suite, len(found))
        assert (np.diff(e) == 1).all(), f"{suite}: episodes not one contiguous range"
        report[suite] = {"tasks": f"{len(found)}/10", "episodes": [int(e[0]), int(e[-1])],
                         "frames": int(data["episode_index"].isin(e).sum())}
    report["total_frames"] = sum(v["frames"] for v in report.values())
    if set(cfg["suites"]) == {"libero_spatial", "libero_object", "libero_goal"}:
        assert report["total_frames"] == 171_996, report["total_frames"]
    # decoder parity: our per-episode decode == LeRobotDataset(return_uint8=True)[i], torchcodec (needs AV1 FFmpeg)
    from huggingface_hub import hf_hub_download
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.import_utils import get_safe_default_video_backend
    assert get_safe_default_video_backend() == "torchcodec", "torchcodec unavailable (system FFmpeg 4-8 with AV1?)"
    key = "observation.images.image"
    hf_hub_download(cfg["dataset"], "videos/observation.images.image2/chunk-000/file-000.mp4", repo_type="dataset",
                    revision=cfg["revision"])   # __getitem__ decodes every camera; episodes in file 0 need this one
    e = eps.index[(eps[f"videos/{key}/chunk_index"] == 0) & (eps[f"videos/{key}/file_index"] == 0)][-1]
    # episodes=[e]: otherwise LeRobotDataset wants every episode's wrist video and downloads ~0.9 GB of them
    ds = LeRobotDataset(cfg["dataset"], root=root, revision=cfg["revision"], episodes=[int(e)], return_uint8=True,
                        video_backend="torchcodec")
    de = data[data["episode_index"] == e]
    r = eps.loc[e]
    path = root / info["video_path"].format(video_key=key, chunk_index=0, file_index=0)
    fr = _decode_episode((str(path), (r[f"videos/{key}/from_timestamp"] + de["timestamp"].to_numpy(np.float64)).tolist()))
    for k in (0, len(fr) // 2, len(fr) - 1):
        assert (ds[k][key].permute(1, 2, 0).numpy() == fr[k]).all(), f"decoder parity failed at ep {e} frame {k}"
    report["decoder_parity_episode"] = int(e)
    log(f"G1 pass: {report}")
    _gate_log(cfg, "G1", report)


def gate_g2(cfg, n=3):
    """First-frame parity: nearest init state gives small error, a 180-degree rotation a large one."""
    report = {}
    for suite in cfg["suites"]:
        fc, init = load_frames(cfg, suite), init_frames(cfg, suite)
        rows = []
        for e in np.unique(fc["ep"])[:: max(1, len(np.unique(fc["ep"])) // n)][:n]:
            k, mse, unrot, serr = match_init(fc, init, e)
            rows.append({"ep": int(e), "init": k, "mse": mse, "mse_unrotated": unrot, "state_err": serr})
            assert mse < 0.25 * unrot, f"G2 fail {suite} ep {e}: mse {mse:.1f} vs unrotated {unrot:.1f}"
            r = np.flatnonzero(fc["ep"] == e)[0]
            err = np.abs(init["state"][fc["task"][r], k] - fc["state"][r])
            # init noise moves the eef ~1-5 cm between demos; a convention mismatch (frame, quat order, 2pi flip,
            # gripper sign/order) between dataset state and LiberoProcessorStep state is far larger
            assert err[:6].max() < 0.1 and err[6:].max() < 0.01, f"G2 fail {suite} ep {e}: state err {err.round(3)}"
        report[suite] = rows
    log(f"G2 pass: {report}")
    _gate_log(cfg, "G2", report)


def gate_g5(cfg, model, res, device, text, n=256):
    """Adapter gate: shapes, finiteness, determinism, rotation sensitivity, prompt (in)variance,
    dataset-vs-env cosine, frames/s, VRAM."""
    fc = load_frames(cfg, cfg["suites"][0])
    rows = np.sort(np.random.default_rng(0).choice(len(fc["ep"]), n, replace=False))
    frames = np.asarray(fc["frames"][rows])
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    enc = load(model, device)
    t0 = time.time()
    a = enc.encode(frames, texts_for(fc, rows, "generic"))
    fps = n / (time.time() - t0)   # closed-loop sites only
    sites = list(enc.sites) + list(enc.offline_sites)   # §1.6 offline sites get the same shape/finite/det checks
    if enc.offline_sites:
        a |= enc.encode(frames, texts_for(fc, rows, "generic"), enc.offline_sites)
    b = enc.encode(frames, texts_for(fc, rows, "generic"), sites)
    rot = enc.encode(np.ascontiguousarray(frames[:, ::-1, ::-1]), texts_for(fc, rows, "generic"), sites)
    rep = {"fps": fps, "sites": {}}
    for s in sites:
        x = a[s]
        assert x.shape[:2] == (n, enc.n_tokens) and x.ndim == 3, (s, x.shape)
        assert torch.isfinite(x).all() and x.abs().max() < 65504, (s, "non-finite or overflows the fp16 caches")
        # per-position cosine after removing each position's mean over frames (fixed noise frame, positional
        # embeddings, sink/outlier channels): a 180-degree rotation is mostly a position swap
        c = x.mean(0, keepdim=True)
        cos = torch.nn.functional.cosine_similarity((x - c).flatten(1), (rot[s] - c).flatten(1)).mean().item()
        rep["sites"][s] = {"D": x.shape[-1], "det_maxdiff": (x - b[s]).abs().max().item() / x.abs().max().item(),
                           "rot_cos": cos}
        assert rep["sites"][s]["det_maxdiff"] < 1e-2, (s, rep["sites"][s])
        # ponytail: rotation gates rollout sites only; an offline site's rot_cos is logged, not asserted
        assert cos < 0.99 or s in enc.offline_sites, f"{s}: tokens insensitive to a 180-degree rotation ({cos:.4f})"
    if enc.text:
        ins = enc.encode(frames, texts_for(fc, rows, "instruction"))
        for s in enc.sites:
            rep["sites"][s]["prompt_maxdiff"] = (a[s] - ins[s]).abs().max().item() / a[s].abs().max().item()
            if model in cfg["prompt_invariant"]:   # bf16 + padding noise only
                assert rep["sites"][s]["prompt_maxdiff"] < 1e-2, (s, rep["sites"][s])
    initp = Path(cfg["cache_dir"]) / "probe" / f"init_{cfg['suites'][0]}.npz"
    if initp.exists():   # dataset first frame vs matched env frame (built by G2)
        init = dict(np.load(initp))
        pairs = [(np.flatnonzero(fc["ep"] == e)[0], match_init(fc, init, e)[0]) for e in np.unique(fc["ep"])[:16]]
        ds = np.stack([fc["frames"][r] for r, _ in pairs])
        ev = np.stack([init["frames"][fc["task"][r], k] for r, k in pairs])
        za, zb = enc.encode(ds, [GENERIC_PROMPT] * len(ds)), enc.encode(ev, [GENERIC_PROMPT] * len(ds))
        for s in enc.sites:
            rep["sites"][s]["data_env_cos"] = torch.nn.functional.cosine_similarity(
                za[s].flatten(1), zb[s].flatten(1)).mean().item()
    if device.startswith("cuda"):
        rep["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
    log(f"G5 pass {model}: {rep}")
    res.log("gates", "G5", rep)


def _gate_log(cfg, gate, report):
    p = Path(cfg["results_dir"]) / "gates.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    d = json.loads(p.read_text()) if p.exists() else {}
    d[gate] = report
    p.write_text(json.dumps(d, indent=1, default=float))
