"""Stage CLI. One process per (model, stage):

    python -m repstudy.run --model vjepa2_1 --stage {calibrate,extract,train,eval,probe}
    python -m repstudy.run --stage {analyze,weightdiff}
    python -m repstudy.run --stage {g0,g1,g2,g3,g7}          # env / data gates
    python -m repstudy.run --model X --stage g5               # adapter gate

Every number in results/summary.md traces back to results/<model>.json written by Results.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

CONFIG = {
    # ---- paths ----
    "cache_dir": "cache",        # frames/, tokens/, probe/ (large; ~200 GB box)
    "results_dir": "results",    # <model>.json, heads/, summary.md, figures
    # ---- data (§1.4) ----
    "dataset": "lerobot/libero",
    "revision": "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4",
    "suites": ["libero_spatial", "libero_object", "libero_goal"],
    "image_size": 256,
    "chunk": 8,                  # 8-step action chunks, executed open-loop
    "proprio_dim": 8,
    "action_dim": 7,
    # ---- evaluation (§1.4): seed s runs initial states 10s..10s+9 of every task ----
    "seeds": [0, 1, 2],
    "n_init": 10,
    "blind_max": 0.30,           # §1.5: blind above this on a suite -> proprio dropped on that suite
    # ---- head (identical for every model; tuned once on vjepa2_1 at G6, then frozen) ----
    "head": {"width": 384, "attn_heads": 8, "hidden": 512, "steps": 20_000, "batch": 256,
             "lr": 3e-4, "weight_decay": 1e-4},
    # ---- readout-site calibration (§1.4) ----
    "calib": {"frac": 0.10, "heldout": 0.2, "steps": 3_000},
    # Ladder rungs use matched sites instead of their own argmin: int = fixed index into enc.sites,
    # str = the index selected for that (child) model.
    "site_rule": {"qwen3vl_2b": 3, "cosmos_reason2": 3, "groot_n17": 3,          # L1: LLM layer 16
                  "wan22": "lingbot_va",                                          # L3
                  "vjepa2_l": "vjepa2_1", "dinov3_l": "cradio4", "qwen3vl_4b": "xiaomi_r1",
                  # ablations read out where their base model does
                  "vjepa2_1_t256": "vjepa2_1", "cradio4_t256": "cradio4", "wan22_512": "wan22"},
    # ---- probes / similarity (§1.6) ----
    "probe": {"n_dataset": 6_000, "n_replay": 8_000, "replay_stride": 2, "pca": 512, "folds": 5,
              "sim_n": 4_000, "knn_k": 10, "n_perm": 100},
    # ---- statistics (§1.7) ----
    "n_boot": 10_000,
    # ---- model lineup in production order (§2.4 step 6); Tier 2 runs only as hours remain ----
    "tier1": ["blind", "random_vit", "vjepa2_1", "cradio4", "qwen3vl_2b", "cosmos_reason2", "groot_n17",
              "lingbot_va", "wan22", "molmoact2", "tipsv2", "lingbot_vision"],
    "tier2": ["vggt_omega", "cosmos3_reason", "cosmos3_gen", "xiaomi_r1", "qwen3vl_4b",   # child before parent (site_rule)
              "dinov3_l", "vjepa2_l", "r3m"],
    # §1.5 ablations: 16x16 token budget (closed loop) for two dense ViTs; Wan2.2 at 512^2 (offline: probe only)
    "ablations": ["vjepa2_1_t256", "cradio4_t256"],
    "offline_only": ["wan22_512"],
    "family": {"vjepa2_1_t256": "ablation", "cradio4_t256": "ablation", "wan22_512": "ablation", "random_vit": "floor", "vjepa2_1": "perception", "cradio4": "perception", "tipsv2": "perception",
               "lingbot_vision": "perception", "qwen3vl_2b": "vlm", "cosmos_reason2": "reasoner",
               "groot_n17": "vla", "molmoact2": "vla", "wan22": "video", "lingbot_va": "world_action",
               "vggt_omega": "perception", "cosmos3_reason": "reasoner", "cosmos3_gen": "video",
               "qwen3vl_4b": "vlm", "xiaomi_r1": "vla", "dinov3_l": "perception", "vjepa2_l": "perception",
               "r3m": "perception"},
    "text_models": ["qwen3vl_2b", "cosmos_reason2", "groot_n17", "molmoact2", "wan22", "lingbot_va",
                    "cosmos3_reason", "cosmos3_gen", "qwen3vl_4b", "xiaomi_r1"],
    # image tokens precede the text under causal attention -> image-position tokens cannot see the prompt,
    # so instruction-on equals generic for these (G5 asserts it); only the video models' tokens depend on text
    "prompt_invariant": ["qwen3vl_2b", "cosmos_reason2", "groot_n17", "molmoact2", "qwen3vl_4b", "xiaomi_r1",
                         "cosmos3_reason"],
    # §1.7 planned contrasts: (label, A, B) -> A - B on paired (suite, task, init) cells
    "contrasts": [("cosmos_reason2 - qwen3vl_2b", ["cosmos_reason2"], ["qwen3vl_2b"]),
                  ("groot_n17 - cosmos_reason2", ["groot_n17"], ["cosmos_reason2"]),
                  ("lingbot_va - wan22", ["lingbot_va"], ["wan22"]),
                  ("non-perception - perception",
                   ["qwen3vl_2b", "cosmos_reason2", "groot_n17", "molmoact2", "wan22", "lingbot_va"],
                   ["vjepa2_1", "cradio4", "tipsv2", "lingbot_vision"])],
    "ladders": [("qwen3vl_2b", "cosmos_reason2"), ("cosmos_reason2", "groot_n17"), ("wan22", "lingbot_va")],
    # ---- cut ladder (§1.9); G7 sets it ----
    "cut_level": 0,
    "closed_loop_secondary": True,   # instruction-on closed loop for the L3 pair
    "no_eval": [],                   # models whose closed loop is cut (probes/similarity kept)
    # ---- compute ----
    "encode_batch": 64,
    "n_workers": 8,
}


def apply_cuts(cfg, blind_success=None):
    """§1.9 cut order. Never cut: blind, random floor, gates."""
    cfg = copy.deepcopy(cfg)
    cl = cfg["cut_level"]
    if cl >= 1:
        cfg["tier2"] = []
    if cl >= 2:   # closed loop for secondary conditions and ablations (their offline readouts stay)
        cfg["closed_loop_secondary"] = False
        cfg["no_eval"] = cfg["no_eval"] + cfg["ablations"]
    if cl >= 3 and blind_success:
        cfg["suites"] = [s for s in cfg["suites"] if s != max(blind_success, key=blind_success.get)]
    if cl >= 4:
        cfg["no_eval"] = cfg["no_eval"] + ["wan22"]
    if cl >= 5:
        cfg["seeds"] = cfg["seeds"][:2]
    return cfg


SCHEMA = {"model", "config", "meta", "gates", "calibrate", "site", "train", "eval", "probe"}


class Results:
    """One JSON per model; atomic flush on every write."""

    def __init__(self, path, model=None, cfg=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        else:
            self.data = {"model": model, "config": cfg, "meta": {"created": time.time()}}
            self._flush()

    def log(self, section, key, value):
        assert section in SCHEMA, f"unknown results section {section!r}"
        self.data.setdefault(section, {})[key] = value
        self._flush()

    def get(self, section, key, default=None):
        return self.data.get(section, {}).get(key, default)

    def _flush(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, default=float))
        tmp.replace(self.path)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def results_for(cfg, model):
    return Results(Path(cfg["results_dir"]) / f"{model}.json", model, cfg)


def blind_success(cfg):
    p = Path(cfg["results_dir"]) / "blind.json"
    if not p.exists():
        return None
    ev = json.loads(p.read_text()).get("eval", {})   # keys: suite|text|p<proprio>; the rule reads p1
    return {s: ev[f"{s}|generic|p1"]["success"] for s in cfg["suites"] if f"{s}|generic|p1" in ev}


def proprio_on(cfg, suite):
    """§1.5 rule fixed in advance: blind > blind_max on a suite -> no proprio for any model there."""
    b = blind_success(cfg) or {}
    return not (suite in b and b[suite] > cfg["blind_max"])


def site_for(cfg, model, enc_sites):
    """Selected readout site: ladder rule if any, else calibration argmin."""
    rule = cfg["site_rule"].get(model)
    if isinstance(rule, int):
        return enc_sites[rule]
    if isinstance(rule, str):
        f = Path(cfg["results_dir"]) / f"{rule}.json"   # read-only: never create the rule model's results file
        child = json.loads(f.read_text()).get("site", {}).get("index") if f.exists() else None
        assert child is not None, f"{model} matches {rule}'s site; calibrate {rule} first"
        return enc_sites[child]
    r = results_for(cfg, model)
    curve = r.get("calibrate", "generic")
    assert curve, f"run --stage calibrate for {model} first"
    return min(curve, key=curve.get)


def main(argv=None):
    stages = ["calibrate", "extract", "train", "eval", "probe", "analyze", "weightdiff",
              "g0", "g1", "g2", "g3", "g5", "g6", "g7"]
    p = argparse.ArgumentParser()
    p.add_argument("--model")
    p.add_argument("--stage", required=True, choices=stages)
    p.add_argument("--suites", nargs="+")
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--text", choices=["generic", "instruction"], default="generic")
    p.add_argument("--cut-level", type=int)
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)

    cfg = copy.deepcopy(CONFIG)
    cl = Path(cfg["results_dir"]) / "cut_level"   # persisted: later stages (and analyze) see the level G7 set
    if a.cut_level is not None:
        cl.parent.mkdir(parents=True, exist_ok=True)
        cl.write_text(str(a.cut_level))
    if cl.exists():
        cfg["cut_level"] = int(cl.read_text())
    cfg = apply_cuts(cfg, blind_success(cfg))
    if a.suites:
        cfg["suites"] = a.suites
    if a.seeds:
        cfg["seeds"] = a.seeds
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    from repstudy import analysis, extract, policy  # noqa: E402  (heavy imports after env vars)

    t0 = time.time()
    if a.stage in ("analyze", "weightdiff", "g0", "g1", "g2", "g3", "g6", "g7"):
        {"analyze": analysis.analyze, "weightdiff": analysis.weight_diff_audit,
         "g0": policy.gate_g0, "g1": extract.gate_g1, "g2": extract.gate_g2,
         "g3": policy.gate_g3, "g6": policy.gate_g6, "g7": policy.gate_g7}[a.stage](cfg)
        log(f"{a.stage} done in {time.time() - t0:.0f}s")
        return

    assert a.model, "--model is required for this stage"
    if a.text == "instruction":
        assert a.model in cfg["text_models"], f"{a.model} takes no text"
    res = results_for(cfg, a.model)
    res.log("meta", f"config_{a.stage}_{a.text}", cfg)   # provenance per stage, not only at file creation
    fn = {"calibrate": extract.calibrate, "extract": extract.extract, "train": policy.train,
          "eval": policy.evaluate, "probe": analysis.probe, "g5": extract.gate_g5}[a.stage]
    fn(cfg, a.model, res, a.device, a.text)
    k = f"time_{a.stage}_{a.text}"   # keep the real (slowest) run: a cached re-run takes seconds
    res.log("meta", k, max(time.time() - t0, res.get("meta", k, 0)))
    log(f"{a.model}/{a.stage}/{a.text} done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
