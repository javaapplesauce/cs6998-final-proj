"""Readout head, chunk dataset, training, LIBERO env I/O, closed-loop evaluation; gates G0, G3, G6, G7.

The head is a measuring instrument, identical for every model:
    Linear(D->384) + LayerNorm -> one learned-query attention pool -> concat(proprio, task one-hot)
    -> MLP -> 8-step action chunk, L1 loss, fixed step count, final checkpoint only.
Token/proprio standardisation lives in the head's buffers, so a checkpoint is self-contained.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from repstudy.run import log, proprio_on

MAX_STEPS = {"libero_spatial": 280, "libero_object": 280, "libero_goal": 300}  # == lerobot TASK_SUITE_MAX_STEPS (G0)
NEW_ROLLOUT = {"lerobot_new_rollout": True}  # reset options; key == lerobot NEW_ROLLOUT_OPTION (G0)


def ekey(suite, text, proprio):
    return f"{suite}|{text}|p{int(proprio)}"


# ---------------------------------------------------------------- head


class Head(nn.Module):
    def __init__(self, D, n_tasks=10, proprio=8, act=7, chunk=8, width=384, attn_heads=8, hidden=512):
        super().__init__()
        self.cfg = dict(D=D, n_tasks=n_tasks, proprio=proprio, act=act, chunk=chunk, width=width,
                        attn_heads=attn_heads, hidden=hidden)
        self.register_buffer("tok_mu", torch.zeros(max(D, 1)))
        self.register_buffer("tok_sd", torch.ones(max(D, 1)))
        self.register_buffer("p_mu", torch.zeros(max(proprio, 1)))
        self.register_buffer("p_sd", torch.ones(max(proprio, 1)))
        if D:
            self.proj = nn.Sequential(nn.Linear(D, width), nn.LayerNorm(width))
            self.query = nn.Parameter(torch.randn(1, 1, width) * width ** -0.5)
            self.pool = nn.MultiheadAttention(width, attn_heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear((width if D else 0) + proprio + n_tasks, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, chunk * act))

    def forward(self, tok, prop, task):
        c = self.cfg
        parts = []
        if c["D"]:
            h = self.proj((tok.float() - self.tok_mu) / self.tok_sd)
            parts.append(self.pool(self.query.expand(len(h), -1, -1), h, h, need_weights=False)[0][:, 0])
        if c["proprio"]:
            parts.append((prop.float() - self.p_mu) / self.p_sd)
        parts.append(F.one_hot(task, c["n_tasks"]).float())
        return self.mlp(torch.cat(parts, -1)).view(-1, c["chunk"], c["act"])


def save_head(head, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": head.cfg, "state": head.state_dict()}, path)


def load_head(path, device):
    ck = torch.load(path, map_location=device, weights_only=True)
    head = Head(**ck["cfg"]).to(device)
    head.load_state_dict(ck["state"], strict=True)
    return head.eval()


def head_tag(cfg):
    """Hash of the head + its training recipe: a G6 retune must never reuse old heads or evals."""
    import hashlib
    import json
    return hashlib.sha1(json.dumps([cfg["head"], cfg["chunk"]], sort_keys=True).encode()).hexdigest()[:8]


def head_path(cfg, model, site, suite, seed, text, proprio):
    site = (site or "none").replace("/", "_")
    return (Path(cfg["results_dir"]) / "heads" / model / site / head_tag(cfg)
            / f"{suite}__{text}__p{int(proprio)}__s{seed}.pt")


def cut_closed_loop(cfg, model, text):
    """§1.9: a cut closed loop also skips its token extraction and head training."""
    return model in cfg["no_eval"] or (text == "instruction" and not cfg["closed_loop_secondary"])


def heads_done(cfg, model, site, suite, text):
    return all(head_path(cfg, model, site, suite, s, text, proprio_on(cfg, suite)).exists() for s in cfg["seeds"])


# ---------------------------------------------------------------- chunk dataset + training


def make_data(cfg, tok, mu, sd, fc, rows, suite, device, proprio=None):
    """Device-resident training table for `rows` of a suite's frame cache. tok: (len(rows),64,D) or None.
    Chunk targets are rows t..t+7 of the same episode, clamped to its last frame (repeat-last padding)."""
    rows = np.asarray(rows)
    ep = fc["ep"][rows]
    assert (np.diff(rows) > 0).all(), "rows must be sorted"
    bounds = np.r_[np.flatnonzero(np.diff(ep)) + 1, len(rows)]
    last = np.repeat(bounds - 1, np.diff(np.r_[0, bounds]))
    tgt = np.minimum(np.arange(len(rows))[:, None] + np.arange(cfg["chunk"]), last[:, None])
    d = {"tok": None, "D": 0}
    if tok is not None:
        t = torch.empty(tok.shape, dtype=torch.float16, device=device)
        for a in range(0, len(tok), 8192):
            t[a:a + 8192] = torch.from_numpy(np.array(tok[a:a + 8192], np.float16)).to(device)
        d.update(tok=t, D=tok.shape[-1], mu=torch.as_tensor(mu), sd=torch.as_tensor(sd))
    st = fc["state"][rows]
    d.update(prop=torch.as_tensor(st, device=device), p_mu=torch.as_tensor(st.mean(0)),
             p_sd=torch.as_tensor(st.std(0) + 1e-6), task=torch.as_tensor(fc["task"][rows], device=device),
             act=torch.as_tensor(fc["action"][rows], device=device), tgt=torch.as_tensor(tgt, device=device),
             proprio=proprio_on(cfg, suite) if proprio is None else proprio)
    return d


def _batch(d, idx):
    tok = d["tok"][idx] if d["tok"] is not None else None
    return (tok, d["prop"][idx], d["task"][idx]), d["act"][d["tgt"][idx]]


def fit(cfg, d, train_idx, seed, steps, device):
    h = cfg["head"]
    torch.manual_seed(seed)
    head = Head(d["D"], 10, cfg["proprio_dim"] if d["proprio"] else 0, cfg["action_dim"], cfg["chunk"],
                h["width"], h["attn_heads"], h["hidden"]).to(device)
    if d["D"]:
        head.tok_mu.copy_(d["mu"]), head.tok_sd.copy_(d["sd"])
    if d["proprio"]:
        head.p_mu.copy_(d["p_mu"]), head.p_sd.copy_(d["p_sd"])
    opt = torch.optim.AdamW(head.parameters(), lr=h["lr"], weight_decay=h["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + np.cos(np.pi * min(s, steps) / steps)))
    g = torch.Generator().manual_seed(seed)
    tr = torch.as_tensor(np.asarray(train_idx))
    head.train()
    for _ in range(steps):
        idx = tr[torch.randint(len(tr), (h["batch"],), generator=g)].to(device)
        x, y = _batch(d, idx)
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16,
                            enabled=torch.device(device).type == "cuda"):
            pred = head(*x)
        loss = F.l1_loss(pred.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return head.eval()


@torch.no_grad()
def chunk_l1(head, d, idx, bs=4096):
    idx = torch.as_tensor(np.asarray(idx))
    tot = 0.0
    for a in range(0, len(idx), bs):
        i = idx[a:a + bs].to(d["prop"].device)
        x, y = _batch(d, i)
        tot += F.l1_loss(head(*x).float(), y, reduction="sum").item()
    return tot / (len(idx) * d["tgt"].shape[1] * d["act"].shape[1])


def train(cfg, model, res, device, text):
    """One head per (suite, seed). Token caches are deleted once a suite's heads exist (train is their only reader:
    eval re-encodes live, probes use their own set); a full Tier-1 cache would be ~445 GB otherwise."""
    from repstudy.extract import load_frames, token_path
    if cut_closed_loop(cfg, model, text):
        return log(f"{model}/{text}: closed loop cut (cut_level {cfg['cut_level']}): no heads")
    site = None if model == "blind" else res.get("site", "name")
    assert model == "blind" or site, f"calibrate {model} first"
    for suite in cfg["suites"]:
        if heads_done(cfg, model, site, suite, text):
            continue
        fc = load_frames(cfg, suite)
        rows = np.arange(len(fc["ep"]))
        tok = mu = sd = None
        if site:
            p = token_path(cfg, model, site, text, suite)
            tok = np.load(p, mmap_mode="r")
            s = np.load(p.with_suffix(".stats.npz"))
            mu, sd = s["mean"], s["std"]
        d = make_data(cfg, tok, mu, sd, fc, rows, suite, device)
        for seed in cfg["seeds"]:
            path = head_path(cfg, model, site, suite, seed, text, d["proprio"])
            if path.exists():
                continue
            t0 = time.time()
            head = fit(cfg, d, rows, seed, cfg["head"]["steps"], device)
            save_head(head, path)
            x = _batch(d, torch.arange(4, device=device))[0]
            assert torch.allclose(head(*x), load_head(path, device)(*x), atol=1e-6)   # strict round-trip
            res.log("train", f"{ekey(suite, text, d['proprio'])}|s{seed}",
                    {"site": site, "train_l1": chunk_l1(head, d, rows[::7]), "seconds": time.time() - t0,
                     "path": str(path)})
            log(f"{model}/{suite}/s{seed}: trained in {time.time() - t0:.0f}s")
        del d, tok
        if site and not cfg.get("keep_tokens"):
            p.unlink()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- LIBERO env I/O (Linux only; lazy imports)


ASSETS_REV = "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"   # lerobot/libero-assets (dataset, ungated)


def libero_setup():
    """Before any `import libero.libero` (which prompts on stdin without a config) and before env workers start:
    (1) pre-download assets to ~/.cache/libero/assets, the only place hf-libero 0.1.4 looks (it ignores the config's
    `assets` key; otherwise every forkserver worker races to download), (2) (re)write the config if stale."""
    import importlib.util
    root = Path(importlib.util.find_spec("libero").submodule_search_locations[0]) / "libero"  # no __init__ side effects
    assets = Path.home() / ".cache" / "libero" / "assets"
    if not all((assets / d).is_dir() for d in ("articulated_objects", "stable_scanned_objects", "turbosquid_objects",
                                               "stable_hope_objects", "scenes", "textures")):
        from huggingface_hub import snapshot_download
        snapshot_download("lerobot/libero-assets", repo_type="dataset", revision=ASSETS_REV, local_dir=assets)
    cfg_path = Path(os.environ.get("LIBERO_CONFIG_PATH", Path.home() / ".libero")) / "config.yaml"
    text = "".join(f"{k}: {v}\n" for k, v in {
        "benchmark_root": root, "bddl_files": root / "bddl_files", "init_states": root / "init_files",
        "datasets": root.parent / "datasets", "assets": assets}.items())
    if not cfg_path.exists() or cfg_path.read_text() != text:
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(text)
    return assets


def suite_obj(suite):
    libero_setup()
    from libero.libero import benchmark
    return benchmark.get_benchmark_dict()[suite]()


def task_names(suite):
    s = suite_obj(suite)
    return [s.get_task(i).language for i in range(s.n_tasks)]


def n_init_states(suite, tid):
    s = suite_obj(suite)   # runs libero_setup before lerobot.envs.libero imports libero.libero
    from lerobot.envs.libero import get_task_init_states
    return len(get_task_init_states(s, tid))


def _env_fn(suite, tid, init):
    libero_setup()   # before `import libero.libero`; AsyncVectorEnv calls this first in the parent (dummy env)
    from lerobot.envs.libero import LiberoEnv
    return LiberoEnv(task_suite=suite_obj(suite), task_id=tid, task_suite_name=suite, camera_name="agentview_image",
                     obs_type="pixels_agent_pos", observation_width=256, observation_height=256,
                     episode_index=init, n_envs=1)


def vec_env(suite, tid, inits):
    """One sub-env per initial state; sub-env i starts at init state inits[i] (exact control per seed)."""
    import gymnasium as gym
    from lerobot.envs.utils import freeze_after_episode_end
    return gym.vector.AsyncVectorEnv([freeze_after_episode_end(partial(_env_fn, suite, tid, i)) for i in inits],
                                     context="forkserver", shared_memory=True,
                                     autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP)


def policy_obs(obs):
    """Raw (batched) LIBERO obs -> (uint8 frames (B,256,256,3) in dataset orientation, float state (B,8)).
    LiberoProcessorStep does the 180-degree rotation and eef pos + axis-angle + gripper qpos."""
    from lerobot.envs.utils import preprocess_observation
    from lerobot.processor import LiberoProcessorStep
    o = LiberoProcessorStep().observation(preprocess_observation(obs))
    img = o["observation.images.image"].permute(0, 2, 3, 1)
    return (img * 255).round().clamp(0, 255).to(torch.uint8).numpy(), o["observation.state"].float()


def _batched(o):
    return {k: _batched(v) for k, v in o.items()} if isinstance(o, dict) else (o[None] if isinstance(o, np.ndarray) else o)


def success_flags(info, n):
    """is_success on the step an episode ends; also read final_info in case of SAME_STEP autoreset."""
    s = np.zeros(n, bool)
    for src in (info, info.get("final_info") if isinstance(info.get("final_info"), dict) else {}):
        if "is_success" in src:
            s |= np.asarray(src["is_success"], bool) & np.asarray(src.get("_is_success", np.ones(n, bool)), bool)
    return s


def reset_frames(suite, tid, inits, chunk=25):
    inits = list(inits)
    fr, st = [], []
    for a in range(0, len(inits), chunk):
        env = vec_env(suite, tid, inits[a:a + chunk])
        obs, _ = env.reset(seed=0, options=NEW_ROLLOUT)
        f, s = policy_obs(obs)
        fr.append(f), st.append(s.numpy())
        env.close()
    return np.concatenate(fr), np.concatenate(st)


def replay(suite, tid, init, actions):
    """Step demo actions from a given init state (num_steps_wait=10 like the OpenVLA no_noops regeneration the
    dataset matches). Frame i is the observation action i was chosen from.
    -> frames (T,256,256,3), state (T,8), target-object world position (T,3), success.
    Target = the task's obj_of_interest[0] (object, fixture or region site; always in object_states_dict because
    LIBERO's own success check reads it). Raw-obs '<obj>_pos' keys would miss fixtures/regions and include the
    arm-relative '<obj>_to_robot0_eef_pos'."""
    env = _env_fn(suite, tid, init)
    obs, _ = env.reset(seed=0)
    target = env._env.obj_of_interest[0]
    fr, st, obj, success = [], [], [], False
    for a in actions:
        f, s = policy_obs(_batched(obs))
        fr.append(f[0]), st.append(s[0].numpy())
        obj.append(np.asarray(env._env.env.object_states_dict[target].get_geom_state()["pos"], np.float32))
        obs, _, term, _, info = env.step(np.asarray(a, np.float64))
        success |= bool(info["is_success"])
        if term:   # == is_success for LiberoEnv
            break
    env.close()
    return np.stack(fr), np.stack(st), np.stack(obj), success


# ---------------------------------------------------------------- closed-loop evaluation


def rollout(env, enc, site, heads, task_id, text, cfg, max_steps, device):
    """Batched open-loop-chunk rollout. heads: [(slice, head)] covering all sub-envs."""
    n = env.num_envs
    obs, _ = env.reset(seed=list(range(n)), options=NEW_ROLLOUT)
    done, succ, steps = np.zeros(n, bool), np.zeros(n, bool), np.full(n, max_steps)
    task = torch.full((n,), task_id, dtype=torch.long, device=device)
    t = 0
    while t < max_steps and not done.all():
        frames, state = policy_obs(obs)
        state = state.to(device)
        tok = enc.encode(frames, [text] * n, [site])[site].to(device) if enc else None
        act = np.zeros((n, cfg["chunk"], cfg["action_dim"]), np.float32)
        with torch.no_grad():
            for sl, head in heads:
                act[sl] = head(tok[sl] if tok is not None else None, state[sl], task[sl]).float().cpu().numpy()
        for k in range(cfg["chunk"]):
            if t >= max_steps or done.all():
                break
            obs, _, term, trunc, info = env.step(act[:, k])
            t += 1
            s = success_flags(info, n)
            ended = ~done & (np.asarray(term) | np.asarray(trunc) | s)
            succ |= ~done & s
            steps[ended] = t
            done |= ended
    return succ, steps


def evaluate(cfg, model, res, device, text, make_env=None):
    from repstudy.encoders import GENERIC_PROMPT, load
    if cut_closed_loop(cfg, model, text):
        return log(f"{model}/{text}: closed loop cut (cut_level {cfg['cut_level']})")
    make_env = make_env or vec_env
    enc = site = None
    if model != "blind":
        enc = load(model, device)
        site = res.get("site", "name")
        assert site in enc.sites, (site, enc.sites)
    n_init, seeds = cfg["n_init"], cfg["seeds"]
    for suite in cfg["suites"]:
        P = proprio_on(cfg, suite)   # blind: True on its first run, so its p1 result decides the rule
        key = ekey(suite, text, P)
        old = res.get("eval", key)
        if old and old["site"] == site and old.get("head") == head_tag(cfg) and set(seeds) <= set(map(int, old["per_seed"])):
            log(f"{model}/{key}: cached")
            continue
        heads = [(slice(j * n_init, (j + 1) * n_init), load_head(head_path(cfg, model, site, suite, s, text, P), device))
                 for j, s in enumerate(seeds)]
        names = task_names(suite) if text == "instruction" else None
        inits = [n_init * s + i for s in seeds for i in range(n_init)]   # seed s -> init states 10s..10s+9
        trials, t0 = [], time.time()
        for tid in range(10):
            env = make_env(suite, tid, inits)
            assert env.single_action_space.shape == (cfg["action_dim"],), env.single_action_space
            prompt = names[tid] if text == "instruction" else GENERIC_PROMPT
            succ, steps = rollout(env, enc, site, heads, tid, prompt, cfg, MAX_STEPS[suite], device)
            env.close()
            trials += [[seeds[i // n_init], tid, inits[i], int(succ[i]), int(steps[i])] for i in range(len(inits))]
            log(f"{model}/{suite}/task{tid}: {succ.mean():.2f}")
        tr = np.array(trials)
        res.log("eval", key, {
            "success": float(tr[:, 3].mean()),
            "per_seed": {int(s): float(tr[tr[:, 0] == s, 3].mean()) for s in seeds},
            "per_task": [float(tr[tr[:, 1] == k, 3].mean()) for k in range(10)],
            "trials": trials, "fields": ["seed", "task", "init", "success", "steps"],
            "site": site, "head": head_tag(cfg), "seconds": time.time() - t0})
        log(f"{model}/{key}: success {tr[:, 3].mean():.3f}")


# ---------------------------------------------------------------- gates


def gate_g0(cfg):
    """venv resolves, imports work, no interactive prompt, assets pre-downloaded, one EGL frame renders."""
    assets = libero_setup()
    code = ("import sklearn, scipy, matplotlib, libero.libero, lerobot.envs.libero, lerobot.processor; "
            "from pathlib import Path; "
            "from libero.libero import get_assets_path; print('ASSETS', Path(get_assets_path()).resolve())")
    r = subprocess.run([sys.executable, "-c", code], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                       timeout=600, env={**os.environ, "MUJOCO_GL": "egl"})
    assert r.returncode == 0, f"G0 import failed / prompted on stdin:\n{r.stdout}\n{r.stderr}"
    assert f"ASSETS {assets.resolve()}" in r.stdout, f"G0: hf-libero would not use {assets}:\n{r.stdout}"
    from lerobot.envs.libero import TASK_SUITE_MAX_STEPS
    from lerobot.envs.utils import NEW_ROLLOUT_OPTION
    assert all(TASK_SUITE_MAX_STEPS[s] == v for s, v in MAX_STEPS.items()), TASK_SUITE_MAX_STEPS
    assert set(NEW_ROLLOUT) == {NEW_ROLLOUT_OPTION}, NEW_ROLLOUT_OPTION
    frames, state = reset_frames(cfg["suites"][0], 0, [0])
    assert frames.shape == (1, 256, 256, 3) and frames.dtype == np.uint8 and frames.std() > 5, frames.shape
    assert state.shape == (1, 8), state.shape
    import lerobot
    log(f"G0 pass: lerobot {lerobot.__version__}, torch {torch.__version__}, cuda {torch.cuda.is_available()}, "
        f"frame mean {frames.mean():.1f}")


def gate_g3(cfg, n=2):
    """Harness sanity: replayed demo actions from the matched init state succeed at least once."""
    from repstudy.extract import init_frames, load_frames, match_init
    report = {}
    for suite in cfg["suites"]:
        fc, init = load_frames(cfg, suite), init_frames(cfg, suite)
        out = []
        for e in np.unique(fc["ep"])[:n]:
            k = match_init(fc, init, e)[0]
            out.append(bool(replay(suite, int(fc["task"][fc["ep"] == e][0]), k, fc["action"][fc["ep"] == e])[3]))
        report[suite] = out
    ok = any(any(v) for v in report.values())
    log(f"G3 {'pass' if ok else 'FAIL'}: replay successes {report}")
    if not ok:
        log("G3 fallback: run stock `lerobot-eval --policy.path=nvidia/gr00t17-lerobot-libero_spatial-640 "
            "--env.type=libero --env.task=libero_spatial` and check it lands near its documented 95%.")
    from repstudy.extract import _gate_log
    _gate_log(cfg, "G3", report)
    assert ok, "G3 failed"


def gate_g6(cfg, suite="libero_spatial", seed=0, margin=0.10, pilot="vjepa2_1"):
    """Pilot (vjepa2_1, one suite/seed) must be clearly above blind (same proprio condition) and not at floor."""
    import json

    from repstudy.extract import _gate_log
    P = proprio_on(cfg, suite)

    def ev(m):
        f = Path(cfg["results_dir"]) / f"{m}.json"
        e = json.loads(f.read_text()).get("eval", {}).get(ekey(suite, "generic", P)) if f.exists() else None
        assert e and str(seed) in e["per_seed"], f"G6: run {m} train+eval on {suite} (p{int(P)}) first"
        assert e.get("head") == head_tag(cfg), f"G6: {m}'s {suite} eval used an older head; re-run train+eval"
        return e["per_seed"][str(seed)]

    v, b = ev(pilot), ev("blind")
    rep = {pilot: v, "blind": b, "proprio": P, "margin": margin, "head": head_tag(cfg)}
    _gate_log(cfg, "G6", rep)
    assert v >= b + margin and v > 0.05, f"G6 fail {rep}: fix the head (CONFIG['head']) before production"
    log(f"G6 pass: {rep}")


PLAN_H = {**dict.fromkeys(["random_vit", "vjepa2_1", "cradio4", "tipsv2", "lingbot_vision", "vjepa2_l", "dinov3_l",
                           "r3m", "vggt_omega"], 0.95),
          **dict.fromkeys(["qwen3vl_2b", "cosmos_reason2", "groot_n17", "qwen3vl_4b", "xiaomi_r1"], 1.15),
          "molmoact2": 1.6, "wan22": 1.55, "lingbot_va": 1.55, "cosmos3_reason": 2.0, "cosmos3_gen": 3.0,
          "blind": 0.4}   # §1.9 block midpoints (h, all three suites, incl. eval); replaced by measurements


def gate_g7(cfg, n_envs=30, n_steps=60):
    """Measured env-steps/s + stage times -> projected hours per cut level -> smallest level within budget."""
    import json

    from repstudy import run
    from repstudy.extract import _gate_log
    env = vec_env(cfg["suites"][0], 0, list(range(n_envs)))
    env.reset(seed=0, options=NEW_ROLLOUT)
    t0 = time.time()
    for _ in range(n_steps):
        env.step(np.zeros((n_envs, cfg["action_dim"]), np.float32))
    sps = n_envs * n_steps / (time.time() - t0)
    env.close()
    # sim-bound closed loop of one model: suites x 10 tasks x n_init x seeds, mean episode ~0.75 x step cap
    eval_h = len(run.CONFIG["suites"]) * 10 * cfg["n_init"] * 3 * 0.75 * np.mean(list(MAX_STEPS.values())) / sps / 3600
    measured = {}
    for f in Path(cfg["results_dir"]).glob("*.json"):
        d = json.loads(f.read_text())
        full = all(ekey(s, "generic", proprio_on(cfg, s)) in d.get("eval", {}) for s in run.CONFIG["suites"])
        if full:   # the 1-suite/1-seed pilot is not a full model measurement
            measured[f.stem] = sum(v for k, v in d.get("meta", {}).items() if k.startswith("time_")) / 3600

    def est(m, c):
        base = m.removesuffix("_t256").removesuffix("_512")
        h = measured.get(m, max(PLAN_H.get(base, 1.5), eval_h)) * (4 if m.endswith("_t256") else 1)
        if m in c["no_eval"]:
            h -= eval_h
        return max(h, 0.1) * len(c["suites"]) / 3 * (1 if len(c["seeds"]) == 3 else 0.8)

    def total(level):
        c = run.apply_cuts({**run.CONFIG, "cut_level": level}, run.blind_success(cfg))
        ms = c["tier1"] + c["tier2"] + c["ablations"] + c["offline_only"]
        sec = 2 * eval_h if c["closed_loop_secondary"] else 0.0   # L3 instruction-on closed loops
        return sum(est(m, c) for m in ms) + sec + 1.0               # + probes/similarity/figures

    budget = float(os.environ.get("REPSTUDY_BUDGET_H", 20))
    proj = {level: total(level) for level in range(6)}
    cut = next((level for level in range(6) if proj[level] <= budget), 5)
    rep = {"env_steps_per_s": sps, "eval_h_per_model": eval_h, "projected_h_by_cut": proj, "budget_h": budget,
           "suggested_cut_level": cut, "measured_h": measured}
    _gate_log(cfg, "G7", rep)
    log(f"G7: suggested --cut-level {cut}: {rep}")
