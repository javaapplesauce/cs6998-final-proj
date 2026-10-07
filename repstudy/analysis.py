"""Representation analysis and reporting (§1.6, §1.7): state probes, null-calibrated similarity,
planned contrasts, probe-success link, weight-diff audit, summary.md + figures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from repstudy.run import log, proprio_on, results_for

# ---------------------------------------------------------------- probes

TARGETS = ("eef_pos", "eef_rot", "gripper", "progress", "obj_pos", "offset")


def probe_targets(L):
    """name -> (Y (M,k), mask). Object targets exist only for replayed frames.
    eef_rot is the rotation matrix: the state's axis-angle wraps near pi, which a linear probe cannot fit."""
    from scipy.spatial.transform import Rotation
    st = L["state"]
    obj = ~np.isnan(L["obj"]).any(1)
    allm = np.ones(len(st), bool)
    rot = Rotation.from_rotvec(st[:, 3:6]).as_matrix().reshape(-1, 9)
    return {"eef_pos": (st[:, :3], allm), "eef_rot": (rot, allm),
            "gripper": (st[:, 6:7] - st[:, 7:8], allm), "progress": (L["progress"][:, None], allm),
            "obj_pos": (L["obj"], obj), "offset": (L["offset"], obj)}


def r2(y, p):
    """Mean over output dims of 1 - SS_res/SS_tot."""
    return float(np.mean(1 - ((y - p) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0).clip(1e-12)))


def probe_scores(X, L, n_comp=512, folds=5):
    """Episode-grouped K-fold: standardise + PCA fit on the train fold, RidgeCV per target (R2),
    logistic regression for task identity (accuracy)."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    X = np.asarray(X, np.float32)
    tg = probe_targets(L)
    pred = {k: np.full_like(Y, np.nan, dtype=np.float64) for k, (Y, _) in tg.items()}
    task_pred = np.zeros(len(X), np.int64)
    for tr, te in GroupKFold(folds).split(X, groups=L["group"]):
        sc = StandardScaler().fit(X[tr])
        pca = PCA(min(n_comp, len(tr) - 1, X.shape[1]), random_state=0).fit(sc.transform(X[tr]))
        ztr, zte = pca.transform(sc.transform(X[tr])), pca.transform(sc.transform(X[te]))
        for k, (Y, m) in tg.items():
            mt = m[tr]
            if mt.sum() > 10 and m[te].any():
                # alpha chosen by episode-grouped inner CV: efficient LOO would leak neighbouring frames of an episode
                inner = list(GroupKFold(3).split(ztr[mt], groups=L["group"][tr][mt]))
                rg = RidgeCV(alphas=np.logspace(-2, 5, 15), cv=inner).fit(ztr[mt], Y[tr][mt])
                pred[k][te[m[te]]] = rg.predict(zte[m[te]])
        task_pred[te] = LogisticRegression(max_iter=2000).fit(ztr, L["task"][tr]).predict(zte)
    out = {k: r2(Y[m], pred[k][m]) for k, (Y, m) in tg.items() if m.any() and not np.isnan(pred[k][m]).any()}
    out["task"] = float((task_pred == L["task"]).mean())
    return out


def pooled(tok):
    """(B,N,D) square row-major grid -> mean (B,D) and 2x2 grid pool (B,4D)."""
    side = int(round(tok.shape[1] ** 0.5))
    g = tok.reshape(len(tok), side, side, -1).permute(0, 3, 1, 2)
    return tok.mean(1), torch.nn.functional.adaptive_avg_pool2d(g, 2).flatten(1)


def rep_dir(cfg, model):
    return Path(cfg["cache_dir"]) / "probe_reps" / model


def probe(cfg, model, res, device, text):
    """Encode the fixed probe set at every site (incl. offline-only sites) and fit probes."""
    from repstudy.encoders import GENERIC_PROMPT, load
    from repstudy.extract import load_probe_set
    if model == "blind":
        return log("blind: nothing to probe")
    frames, L = load_probe_set(cfg)
    enc = load(model, device)
    sites = list(enc.sites) + list(enc.offline_sites)
    d = rep_dir(cfg, model)
    if all(s in res.data.get("probe", {}) and (d / f"{s.replace('/', '_')}.npz").exists() for s in sites):
        return log(f"{model}: probes cached")
    d.mkdir(parents=True, exist_ok=True)
    mean, grid = {s: [] for s in sites}, {s: [] for s in sites}
    for a in range(0, len(frames), cfg["encode_batch"]):
        z = enc.encode(np.asarray(frames[a:a + cfg["encode_batch"]]), [GENERIC_PROMPT] * len(frames[a:a + cfg["encode_batch"]]), sites)
        for s, v in z.items():
            m, g = pooled(v)
            mean[s].append(m.numpy()), grid[s].append(g.half().numpy())
    for s in sites:
        X = np.concatenate(grid[s])
        np.savez(d / f"{s.replace('/', '_')}.npz", mean=np.concatenate(mean[s]), grid2=X)
        sc = probe_scores(X, L, cfg["probe"]["pca"], cfg["probe"]["folds"])
        res.log("probe", s, sc)
        log(f"{model}/{s}: " + ", ".join(f"{k}={v:.3f}" for k, v in sc.items()))


# ---------------------------------------------------------------- similarity (null-calibrated)


def _center_gram(X):
    X = X - X.mean(0, keepdim=True)
    K = X @ X.T
    return K - K.mean(0, keepdim=True) - K.mean(1, keepdim=True) + K.mean()


def _knn(X, k):
    Xn = torch.nn.functional.normalize(X - X.mean(0), dim=1)
    S = Xn @ Xn.T
    S.fill_diagonal_(-float("inf"))
    return S.topk(k, dim=1).indices


def _calib(raw, null):
    mu = float(null.mean())
    return {"raw": float(raw), "null_mean": mu, "null_q95": float(np.quantile(null, 0.95)),
            "calibrated": (float(raw) - mu) / max(1e-12, 1 - mu),
            "p": float((1 + (null >= raw).sum()) / (1 + len(null)))}


def similarity(reps, k=10, n_perm=100, seed=0, device="cpu"):
    """reps: {name: (n,d) array, same rows}, z-scored per dim. -> {'cka': {a: {b: calib}}, 'mknn': {...}}.
    Null: permute the rows of one representation (destroys pairing, keeps marginal geometry)."""
    names = list(reps)
    # z-score per dimension: LLM residual streams carry huge outlier dims that would dominate CKA and cosine kNN
    X = {m: torch.as_tensor(np.asarray(v, np.float32), device=device) for m, v in reps.items()}
    X = {m: (x - x.mean(0)) / (x.std(0) + 1e-6) for m, x in X.items()}
    G = {m: _center_gram(x) for m, x in X.items()}
    Gn = {m: g.norm() for m, g in G.items()}
    NN = {m: _knn(x, k) for m, x in X.items()}
    n = len(next(iter(X.values())))
    gen = torch.Generator().manual_seed(seed)
    perms = [torch.randperm(n, generator=gen).to(device) for _ in range(n_perm)]
    out = {"cka": {a: {} for a in names}, "mknn": {a: {} for a in names}}
    for i, a in enumerate(names):
        for b in names[i:]:
            ga, gb = G[a], G[b]
            cka = lambda p=None: ((ga * (gb if p is None else gb[p][:, p])).sum() / (Gn[a] * Gn[b])).item()  # noqa: E731

            def mknn(p=None):
                nb = NN[b] if p is None else torch.argsort(p)[NN[b][p]]
                return (NN[a][:, :, None] == nb[:, None, :]).any(-1).float().sum(1).mean().item() / k

            for name, f in (("cka", cka), ("mknn", mknn)):
                c = _calib(f(), np.array([f(p) for p in perms]))
                out[name][a][b] = out[name][b][a] = c
    return out


# ---------------------------------------------------------------- statistics


def cell_matrix(cfg, model, text="generic"):
    """-> success (n_clusters=(suite,task), n_init) over cfg suites and exactly cfg seeds' init states, or None if any
    suite or cell is missing. Read-only (never creates a results file); rows are paired across models by init id."""
    p = Path(cfg["results_dir"]) / f"{model}.json"
    if not p.exists():
        return None
    ev = json.loads(p.read_text()).get("eval", {})
    want = np.array(sorted(cfg["n_init"] * s + i for s in cfg["seeds"] for i in range(cfg["n_init"])))
    rows = []
    for suite in cfg["suites"]:
        e = ev.get(f"{suite}|{text}|p{int(proprio_on(cfg, suite))}")
        if e is None:
            return None
        tr = np.array(e["trials"])
        tr = tr[np.isin(tr[:, 0], cfg["seeds"])]
        for task in range(10):
            t = tr[tr[:, 1] == task]
            t = t[np.argsort(t[:, 2])]
            if not np.array_equal(t[:, 2], want):
                return None
            rows.append(t[:, 3])
    return np.stack(rows).astype(np.float64)


def cluster_boot(arr, n_boot=10_000, seed=0):
    """Cluster bootstrap over (suite,task): resample clusters and keep each cluster's mean over its init states
    (that mean already carries the within-cluster variance; resampling inits again would count it twice).
    arr: (C,I) -> bootstrap distribution of the overall mean (n_boot,)."""
    m = arr.mean(1)
    return m[np.random.default_rng(seed).integers(len(m), size=(n_boot, len(m)))].mean(1)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        adj[i] = run
    return adj


def contrasts(cfg):
    """§1.7 planned contrasts on paired (suite, task, init) cells; Holm across the four."""
    rows = []
    for label, A, B in cfg["contrasts"]:
        ma = [m for m in A if cell_matrix(cfg, m) is not None]
        mb = [m for m in B if cell_matrix(cfg, m) is not None]
        if not ma or not mb:
            rows.append({"contrast": label, "missing": sorted(set(A + B) - set(ma + mb))})
            continue
        diff = np.mean([cell_matrix(cfg, m) for m in ma], 0) - np.mean([cell_matrix(cfg, m) for m in mb], 0)
        bs = cluster_boot(diff, cfg["n_boot"])
        rows.append({"contrast": label, "delta": float(diff.mean()), "ci95": np.quantile(bs, [0.025, 0.975]).tolist(),
                     "p": float(min(1.0, 2 * (min((bs <= 0).sum(), (bs >= 0).sum()) + 1) / (len(bs) + 1))),
                     "A": ma, "B": mb, "missing": sorted(set(A + B) - set(ma + mb))})
    ok = [r for r in rows if "p" in r]
    for r, a in zip(ok, holm([r["p"] for r in ok])):
        r["p_holm"] = float(a)
    return rows


# ---------------------------------------------------------------- weight-diff audit (CPU only)


def _group(n):
    """Per layer/block group (depth profile), else module path without the parameter name."""
    import re
    m = re.match(r"(.*?(?:layers|blocks|deepstack_merger_list)\.\d+)\.", n)
    return m.group(1) if m else ".".join(n.split(".")[:-1][:3]) or n


def weight_diff(parent_files, child_files, to_parent):
    """Stream two safetensors checkpoints; relative Frobenius change ||W_c - W_p|| / ||W_p|| per group.
    to_parent(child_name) -> parent name or None (child-only). dtype_floor = change from rounding the parent to the
    child's dtype alone (Wan F32 -> LingBot BF16 shows ~1e-3 with no training at all)."""
    from collections import defaultdict
    from contextlib import ExitStack

    from safetensors import safe_open
    num, den, flo = defaultdict(float), defaultdict(float), defaultdict(float)
    child_only, mismatch, used = [], [], set()
    with ExitStack() as st:
        def keys(files):
            out = {}
            for f in files:
                h = st.enter_context(safe_open(f, "pt"))
                out.update(dict.fromkeys(h.keys(), h))
            return out

        P, C = keys(parent_files), keys(child_files)
        for n, hc in C.items():
            pn = to_parent(n)
            if pn is None or pn not in P:
                child_only.append(n)
                continue
            p, c = P[pn].get_tensor(pn), hc.get_tensor(n)
            if p.numel() != c.numel():
                mismatch.append(n)
                continue
            used.add(pn)
            pf, cf = p.float().reshape(c.shape), c.float()   # reshape: Wan Conv3d patch_embedding -> LingBot Linear
            g = "lm_head" if n.endswith("lm_head.weight") else _group(pn)   # keep lm_head apart from embed_tokens
            num[g] += (cf - pf).pow(2).sum().item()
            den[g] += pf.pow(2).sum().item()
            if p.dtype != c.dtype:
                flo[g] += (pf.to(c.dtype).float() - pf).pow(2).sum().item()
    return {"rel": {g: (num[g] / max(den[g], 1e-30)) ** 0.5 for g in num},
            "dtype_floor": {g: (flo[g] / max(den[g], 1e-30)) ** 0.5 for g in flo},
            "child_only": child_only, "shape_mismatch": mismatch, "parent_only": sorted(set(P) - used)}


def _cosmos_to_qwen(n):   # Qwen3-VL-2B ties lm_head to embed_tokens and stores no lm_head; Cosmos/GR00T store one
    return "model.language_model.embed_tokens.weight" if n == "lm_head.weight" else n


def _groot_to_cosmos(n):  # action_head.* is child-only; Cosmos LLM layers 16-27 land in parent_only
    return n.removeprefix("backbone.model.") if n.startswith("backbone.model.") else None


LADDER = {  # child -> (repo, files to fetch, child->parent name map); 839 LingBot = 823 Wan + 2 reshaped + 14 new
    "qwen3vl_2b": ("Qwen/Qwen3-VL-2B-Instruct", ["*.safetensors"], None),
    "cosmos_reason2": ("nvidia/Cosmos-Reason2-2B", ["*.safetensors"], _cosmos_to_qwen),
    "groot_n17": ("nvidia/GR00T-N1.7-3B", ["model-*.safetensors"], _groot_to_cosmos),
    "wan22": ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", ["transformer/*.safetensors"], None),
    "lingbot_va": ("lerobot/lingbot_va_base", ["model.safetensors"],   # '*' would also grab the postprocessor file
                   lambda n: n.removeprefix("transformer.").replace("patch_embedding_mlp.", "patch_embedding.")),
}


def weight_diff_audit(cfg):
    """CPU-only, per ladder step in cfg['ladders']. Note: GR00T's norm/lm_head are unused by its action path."""
    from huggingface_hub import snapshot_download

    def files(m):
        repo, pat, _ = LADDER[m]
        return sorted(str(p) for p in Path(snapshot_download(repo, allow_patterns=pat)).rglob("*.safetensors"))

    out = {}
    for parent, child in cfg["ladders"]:
        out[f"{parent}->{child}"] = r = weight_diff(files(parent), files(child), LADDER[child][2])
        log(f"weight diff {parent}->{child}: {len(r['rel'])} groups, {len(r['child_only'])} child-only tensors")
    (Path(cfg["results_dir"]) / "weightdiff.json").write_text(json.dumps(out, indent=1))


# ---------------------------------------------------------------- analyze: tables + figures


def _site(cfg, R, m):
    """Selected site; offline-only/ablation models inherit their base model's via site_rule."""
    s = R[m].get("site", {}).get("name")
    rule = cfg["site_rule"].get(m)
    return s or (R[rule].get("site", {}).get("name") if isinstance(rule, str) and rule in R else None)


def _models(cfg):
    order = cfg["tier1"] + cfg["tier2"] + cfg["ablations"] + cfg["offline_only"]
    return [m for m in order if (Path(cfg["results_dir"]) / f"{m}.json").exists()]


def analyze(cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    out_dir = Path(cfg["results_dir"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = _models(cfg)
    R = {m: results_for(cfg, m).data for m in models}
    lines = ["# Representation study — results", ""]

    # ---- success (per suite, CI, seed SD, above blind)
    lines += ["## Closed-loop success (generic prompt)", "",
              "| model | family | site | " + " | ".join(cfg["suites"]) + " | mean | above blind |",
              "|---|---|---|" + "---:|" * (len(cfg["suites"]) + 2)]
    success = {}
    blind = cell_matrix(cfg, "blind") if "blind" in models else None
    for m in models:
        cells = cell_matrix(cfg, m)
        if cells is None:
            continue
        per, k = [], 0
        for suite in cfg["suites"]:
            e = R[m]["eval"][f"{suite}|generic|p{int(proprio_on(cfg, suite))}"]
            c = cells[k:k + 10]
            k += 10
            ci = np.quantile(cluster_boot(c, cfg["n_boot"] // 5), [0.025, 0.975])
            sv = [v for k_, v in e["per_seed"].items() if int(k_) in cfg["seeds"]]
            sd = np.std(sv, ddof=1) if len(sv) > 1 else float("nan")
            per.append(f"{c.mean():.3f} [{ci[0]:.2f},{ci[1]:.2f}] ±{sd:.3f}")
        success[m] = float(cells.mean())
        above = f"{cells.mean() - blind.mean():+.3f}" if blind is not None else "n/a"
        lines.append(f"| {m} | {cfg['family'].get(m, '-')} | {R[m].get('site', {}).get('name', '-')} | "
                     + " | ".join(per) + f" | {cells.mean():.3f} | {above} |")
    lines += ["", "Cells: mean [95% cluster-bootstrap CI over (suite,task)] ±seed SD.", ""]

    # ---- planned contrasts
    cs = contrasts(cfg)
    lines += ["## Planned contrasts (paired cluster bootstrap, Holm)", "", "| contrast | Δ | 95% CI | p | p_holm |",
              "|---|---:|---|---:|---:|"]
    for r in cs:
        if "p" in r:
            lab = r["contrast"] + (f" (without {', '.join(r['missing'])})" if r["missing"] else "")
            lines.append(f"| {lab} | {r['delta']:+.3f} | [{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] | "
                         f"{r['p']:.4f} | {r['p_holm']:.4f} |")
        else:
            lines.append(f"| {r['contrast']} | missing: {', '.join(r['missing'])} | | | |")

    # ---- probes at the selected site
    lines += ["", "## State probes at the selected site (R², task = accuracy)", "",
              "| model | " + " | ".join(TARGETS + ("task",)) + " |", "|---|" + "---:|" * (len(TARGETS) + 1)]
    probe_sel = {}
    for m in models:
        s = _site(cfg, R, m)
        p = R[m].get("probe", {}).get(s or "", None)
        if p:
            probe_sel[m] = p
            lines.append(f"| {m} | " + " | ".join(f"{p.get(t, float('nan')):.3f}" for t in TARGETS + ("task",)) + " |")

    # ---- H3: observed vs predicted-future tokens (video models; Cosmos 3 uses O/P site names)
    pairs_h3 = []
    for m in models:
        c = R[m].get("calibrate", {}).get("generic") or {}
        pr = R[m].get("probe", {})
        names = set(c) | set(pr)
        for o in sorted(names, key=lambda x: int("".join(ch for ch in x if ch.isdigit()) or 0)):
            q = o.replace("/obs", "/pred") if o.endswith("/obs") else "P" + o[1:] if o[:1] == "O" and o[1:].isdigit() else None
            if q and q in names:
                f = lambda d, k: f"{d[k]:.4f}" if isinstance(d.get(k), float) else "–"  # noqa: E731
                pairs_h3.append(f"| {m} | {o.split('/')[0] if '/' in o else o[1:]} | {f(c, o)} | {f(c, q)} | "
                                f"{f(pr.get(o, {}), 'offset')} | {f(pr.get(q, {}), 'offset')} |")
    if pairs_h3:
        lines += ["", "## Observed vs predicted-future tokens (H3; held-out chunk L1, lower is better; offset R²)", "",
                  "| model | block | L1 obs | L1 pred | offset R² obs | offset R² pred |", "|---|---|---:|---:|---:|---:|"]
        lines += pairs_h3

    # ---- probe-success link (exploratory, §1.6)
    lines += ["", "## Probe → success link (Spearman across models, exploratory)", ""]
    both = [m for m in probe_sel if m in success and m not in cfg["ablations"] + cfg["offline_only"]]
    link = {}
    for t in TARGETS + ("task",):
        xs = [(probe_sel[m].get(t), success[m]) for m in both if probe_sel[m].get(t) is not None]
        if len(xs) >= 4:
            rho, p = spearmanr(*zip(*xs))
            link[t] = {"rho": float(rho), "p": float(p), "n": len(xs)}
            lines.append(f"- {t}: ρ={rho:+.2f} (p={p:.3f}, n={len(xs)})")

    # ---- similarity
    reps, rng = {}, np.random.default_rng(0)
    for m in models:
        for f in sorted(rep_dir(cfg, m).glob("*.npz")):
            reps[f"{m}:{f.stem}"] = f
    sim, sel = {}, []
    if len(reps) >= 2:
        n_all = len(np.load(next(iter(reps.values())))["mean"])
        rows = np.sort(rng.choice(n_all, min(cfg["probe"]["sim_n"], n_all), replace=False))
        sim = similarity({k: np.load(f)["mean"][rows] for k, f in reps.items()}, cfg["probe"]["knn_k"],
                         cfg["probe"]["n_perm"], device=device)
        sel = [f"{m}:{_site(cfg, R, m).replace('/', '_')}" for m in models
               if _site(cfg, R, m) and f"{m}:{_site(cfg, R, m).replace('/', '_')}" in reps]
        lines += ["", "## Similarity at selected sites (null-calibrated; raw in brackets)", ""]
        for metric in ("cka", "mknn"):
            lines += [f"### {metric}", "", "| | " + " | ".join(s.split(':')[0] for s in sel) + " |",
                      "|---|" + "---:|" * len(sel)]
            for a in sel:
                lines.append(f"| {a.split(':')[0]} | " + " | ".join(
                    f"{sim[metric][a][b]['calibrated']:.2f} ({sim[metric][a][b]['raw']:.2f})" for b in sel) + " |")
            lines.append("")

    # ---- weight diff + gates + sites
    wd = out_dir / "weightdiff.json"
    if wd.exists():
        lines += ["## Weight-diff audit (relative Frobenius change, median over module groups)", ""]
        for k, v in json.loads(wd.read_text()).items():
            ch = list(v["rel"].values())
            floor = f", bf16 rounding floor {np.median(list(v['dtype_floor'].values())):.4f}" if v["dtype_floor"] else ""
            lines.append(f"- {k}: median {np.median(ch):.4f}, max {np.max(ch):.4f}{floor}, "
                         f"{len(v['child_only'])} child-only / {len(v['parent_only'])} parent-only tensors")
    lines += ["", "## Readout sites (held-out chunk L1 per candidate site)", ""]
    for m in models:
        c = R[m].get("calibrate", {}).get("generic")
        if c:
            lines.append(f"- {m}: " + ", ".join(f"{s}={v:.4f}" for s, v in c.items())
                         + f" → **{R[m]['site']['name']}** ({R[m]['site'].get('rule')})")
    lines += ["", "## Secondary condition: instruction-on (§1.4)", ""]
    for m in models:
        c = R[m].get("calibrate", {}).get("instruction")
        if c and "same_as" in c:
            lines.append(f"- {m}: equals generic ({c['why']})")
        elif c:
            g = R[m].get("calibrate", {}).get("generic", {})
            lines.append(f"- {m} offline chunk L1: " + ", ".join(f"{s}={v:.4f} (generic {g.get(s, float('nan')):.4f})"
                                                              for s, v in c.items()))
        ci = cell_matrix(cfg, m, text="instruction")
        if ci is not None:
            lines.append(f"- {m} closed loop: {ci.mean():.3f} (generic {success.get(m, float('nan')):.3f})")
    if "wan22_512" in R:
        lines += ["", "## Resolution check: Wan2.2 at 256² vs 512² (§1.5, probes at the selected site)", ""]
        for m in ("wan22", "wan22_512"):
            pr = R.get(m, {}).get("probe", {}).get(_site(cfg, R, m) or "", {})
            if pr:
                lines.append(f"- {m}: " + ", ".join(f"{k}={v:.3f}" for k, v in pr.items()))
    lines += ["", f"cut_level={cfg['cut_level']}, seeds={cfg['seeds']}, suites={cfg['suites']}", ""]
    (out_dir / "summary.md").write_text("\n".join(lines))
    (out_dir / "analysis.json").write_text(json.dumps(
        {"success": success, "contrasts": cs, "probe_selected": probe_sel, "link": link,
         "similarity": sim, "selected_reps": sel}, indent=1, default=float))

    # ---- figures
    if success:
        fig, ax = plt.subplots(figsize=(max(6, len(success) * 0.7), 3.5))
        base = blind.mean() if blind is not None else 0.0
        ax.bar(range(len(success)), [v - base for v in success.values()])
        ax.set_xticks(range(len(success)), list(success), rotation=45, ha="right")
        ax.set_ylabel("success − blind" if blind is not None else "success (blind missing)"), ax.axhline(0, color="k", lw=0.5)
        fig.tight_layout(), fig.savefig(out_dir / "fig_success.png", dpi=150), plt.close(fig)
    if probe_sel:
        M = np.array([[probe_sel[m].get(t, np.nan) for t in TARGETS + ("task",)] for m in probe_sel])
        fig, ax = plt.subplots(figsize=(7, 0.4 * len(M) + 1.5))
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(M.shape[1]), TARGETS + ("task",), rotation=45, ha="right")
        ax.set_yticks(range(len(M)), list(probe_sel)), fig.colorbar(im)
        fig.tight_layout(), fig.savefig(out_dir / "fig_probes.png", dpi=150), plt.close(fig)
    if len(sel) >= 3:
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform
        from sklearn.manifold import MDS
        C = np.array([[sim["cka"][a][b]["calibrated"] for b in sel] for a in sel])
        Dm = np.clip(1 - (C + C.T) / 2, 0, None)
        np.fill_diagonal(Dm, 0)
        fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
        axs[0].imshow(C, vmin=0, vmax=1), axs[0].set_title("calibrated CKA")
        axs[0].set_xticks(range(len(sel)), [s.split(":")[0] for s in sel], rotation=90)
        axs[0].set_yticks(range(len(sel)), [s.split(":")[0] for s in sel])
        dendrogram(linkage(squareform(Dm, checks=False), "average"), labels=[s.split(":")[0] for s in sel],
                   ax=axs[1], leaf_rotation=90)
        xy = MDS(2, dissimilarity="precomputed", random_state=0).fit_transform(Dm)
        fams = [cfg["family"].get(s.split(":")[0], "-") for s in sel]
        for f in sorted(set(fams)):
            i = [j for j, x in enumerate(fams) if x == f]
            axs[2].scatter(xy[i, 0], xy[i, 1], label=f)
        for j, s in enumerate(sel):
            axs[2].annotate(s.split(":")[0], xy[j], fontsize=7)
        axs[2].legend(fontsize=7), axs[2].set_title("MDS of 1 − calibrated CKA")
        fig.tight_layout(), fig.savefig(out_dir / "fig_similarity.png", dpi=150), plt.close(fig)
    log(f"summary -> {out_dir / 'summary.md'}")
