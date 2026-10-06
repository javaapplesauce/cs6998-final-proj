"""G4: CPU end-to-end with the stub encoder and a stub vector env; never imports lerobot/libero.

Covers the v1 bug classes (strict checkpoint round-trip, action dim asserted against the env, one results
schema) and analysis known-answers (identical representations score maximal; wide independent Gaussians
calibrate to ~0; a planted linear signal is recovered).   Run: pytest tests/test_smoke.py  (< 60 s)
"""
import copy
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repstudy import analysis, encoders, policy, run  # noqa: E402

N_EP, T = 2, 10   # episodes per task, frames per episode


class StubVecEnv:
    """Sub-env i succeeds at step 3 if i is even, fails (terminates unsuccessfully) at step 6 otherwise."""

    def __init__(self, n, act_dim=7):
        self.num_envs, self.t = n, 0
        self.single_action_space = types.SimpleNamespace(shape=(act_dim,))

    def _obs(self):
        rng = np.random.default_rng(self.t)
        return {"frames": rng.integers(0, 255, (self.num_envs, 256, 256, 3), dtype=np.uint8),
                "state": rng.standard_normal((self.num_envs, 8)).astype(np.float32)}

    def reset(self, seed=None, options=None):
        self.t = 0
        return self._obs(), {}

    def step(self, a):
        assert a.shape == (self.num_envs, 7), a.shape
        self.t += 1
        even = np.arange(self.num_envs) % 2 == 0
        succ = even & (self.t == 3)
        term = succ | (~even & (self.t == 6))
        return self._obs(), np.zeros(self.num_envs), term, np.zeros(self.num_envs, bool), \
            {"is_success": succ, "_is_success": np.ones(self.num_envs, bool)}

    def close(self):
        pass


def _write_frame_cache(cfg, suite):
    d = Path(cfg["cache_dir"]) / "frames"
    d.mkdir(parents=True)
    n = 10 * N_EP * T
    rng = np.random.default_rng(0)
    task = np.repeat(np.arange(10), N_EP * T)
    frames = np.lib.format.open_memmap(d / f"{suite}.npy", mode="w+", dtype=np.uint8, shape=(n, 256, 256, 3))
    frames[:] = rng.integers(0, 255, (n, 256, 256, 3), dtype=np.uint8)
    frames[:, :64, :64] = (task * 25)[:, None, None, None]   # task-dependent patch
    frames.flush()
    np.savez(d / f"{suite}.npz", ep=np.repeat(np.arange(10 * N_EP), T), t=np.tile(np.arange(T), 10 * N_EP),
             task=task, state=rng.standard_normal((n, 8)).astype(np.float32),
             action=np.tanh(rng.standard_normal((n, 7))).astype(np.float32), tasks=np.array([f"task {i}" for i in range(10)]))


def _write_probe_set(cfg, m=120):
    d = Path(cfg["cache_dir"]) / "probe"
    d.mkdir(parents=True)
    rng = np.random.default_rng(1)
    np.save(d / "frames.npy", rng.integers(0, 255, (m, 256, 256, 3), dtype=np.uint8))
    obj = rng.standard_normal((m, 3)).astype(np.float32)
    obj[: m // 2] = np.nan
    state = rng.standard_normal((m, 8)).astype(np.float32)
    np.savez(d / "labels.npz", source=(np.arange(m) >= m // 2).astype(np.int64), task=np.arange(m) % 10,
             group=np.arange(m) // 4, progress=rng.random(m).astype(np.float32), state=state, obj=obj,
             offset=obj - state[:, :3])


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    c = copy.deepcopy(run.CONFIG)
    c.update(cache_dir=str(tmp_path / "cache"), results_dir=str(tmp_path / "results"), suites=["libero_spatial"],
             seeds=[0, 1], n_init=2, encode_batch=16, n_boot=200, site_rule={}, tier1=["blind", "stub", "stub2"], tier2=[],
             blind_max=0.6)   # stub-env blind scores 0.5; the rule itself is tested below
    c["head"] = {**c["head"], "steps": 30, "batch": 32}
    c["calib"] = {**c["calib"], "steps": 20, "frac": 0.5}
    c["probe"] = {**c["probe"], "pca": 16, "folds": 3, "sim_n": 60, "n_perm": 10}
    c["contrasts"] = [("stub - stub2", ["stub"], ["stub2"]), ("needs missing", ["groot_n17"], ["stub"])]
    _write_frame_cache(c, "libero_spatial")
    _write_probe_set(c)
    monkeypatch.setattr(run, "CONFIG", c)
    monkeypatch.setattr(policy, "vec_env", lambda suite, tid, inits: StubVecEnv(len(inits)))
    monkeypatch.setattr(policy, "policy_obs", lambda o: (o["frames"], torch.as_tensor(o["state"])))
    monkeypatch.setattr(policy, "task_names", lambda suite: [f"task {i}" for i in range(10)])
    monkeypatch.setitem(encoders.ADAPTERS, "stub2", lambda device: encoders.Encoder(
        "stub2", ["s0", "s1"], encoders._stub(device, d=16).fn))
    return c


def test_stub_encoder_contract():
    enc = encoders.load("stub", "cpu")
    x = np.random.default_rng(0).integers(0, 255, (5, 256, 256, 3), dtype=np.uint8)
    a, b = enc.encode(x), enc.encode(x)
    r = enc.encode(np.ascontiguousarray(x[:, ::-1, ::-1]))
    for s in enc.sites:
        assert a[s].shape == (5, 64, 32) and a[s].dtype == torch.float32
        assert torch.equal(a[s], b[s])
        assert not torch.allclose(a[s], r[s])
    assert encoders.pool8(torch.ones(2, 24 * 24, 3), 24, 24).shape == (2, 64, 3)


def test_end_to_end(cfg, tmp_path):
    for m in ("blind", "stub", "stub2"):
        for stage in ("calibrate", "extract", "train", "eval", "probe"):
            run.main(["--model", m, "--stage", stage, "--device", "cpu"])
    run.main(["--stage", "analyze"])
    res = {m: json.loads((tmp_path / "results" / f"{m}.json").read_text()) for m in ("blind", "stub", "stub2")}

    # one results schema
    for r in res.values():
        assert set(r) <= run.SCHEMA, set(r) - run.SCHEMA
    with pytest.raises(AssertionError):
        run.results_for(cfg, "stub").log("bogus", "k", 1)

    # stages produced what downstream reads
    assert res["stub"]["site"]["name"] in ("s0", "s1")
    ev = res["stub"]["eval"]["libero_spatial|generic|p1"]
    assert len(ev["trials"]) == 10 * 2 * 2 and ev["success"] == pytest.approx(0.5)   # even sub-envs succeed
    assert [t[2] for t in ev["trials"][:4]] == [0, 1, 2, 3]                             # seed s -> inits 2s..2s+1
    assert res["blind"]["eval"]["libero_spatial|generic|p1"]["success"] == pytest.approx(0.5)
    assert set(res["stub"]["probe"]) == {"s0", "s1"} and "task" in res["stub"]["probe"]["s0"]
    summary = (tmp_path / "results" / "summary.md").read_text()
    for h in ("Closed-loop success", "Planned contrasts", "State probes", "Similarity", "missing: groot_n17"):
        assert h in summary, h

    # strict checkpoint round-trip
    p = policy.head_path(cfg, "stub", res["stub"]["site"]["name"], "libero_spatial", 0, "generic", True)
    head = policy.load_head(p, "cpu")
    with pytest.raises(RuntimeError):
        policy.Head(**{**head.cfg, "hidden": 64}).load_state_dict(torch.load(p, weights_only=True)["state"], strict=True)

    # action dim asserted against the env
    cfg_bad = copy.deepcopy(cfg)
    r = run.Results(tmp_path / "bad.json", "stub", cfg_bad)
    r.log("site", "name", res["stub"]["site"]["name"])
    with pytest.raises(AssertionError):
        policy.evaluate(cfg_bad, "stub", r, "cpu", "generic", make_env=lambda s, t, i: StubVecEnv(len(i), act_dim=6))


def test_blind_rule_drops_proprio(cfg):
    r = run.results_for(cfg, "blind")
    r.log("eval", "libero_spatial|generic|p1", {"success": 0.9})
    assert not run.proprio_on(run.CONFIG, "libero_spatial")
    assert run.apply_cuts({**cfg, "cut_level": 3}, run.blind_success(cfg))["suites"] == []


def test_similarity_known_answers():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 30))
    same = analysis.similarity({"a": X, "b": X}, k=10, n_perm=20)
    assert same["cka"]["a"]["b"]["raw"] == pytest.approx(1, abs=1e-5)
    assert same["mknn"]["a"]["b"]["raw"] == pytest.approx(1)
    assert same["cka"]["a"]["b"]["calibrated"] > 0.95 and same["cka"]["a"]["b"]["p"] < 0.1
    wide = analysis.similarity({"a": rng.standard_normal((200, 2000)), "b": rng.standard_normal((200, 2000))},
                               k=10, n_perm=50)
    assert wide["cka"]["a"]["b"]["raw"] > 0.3                      # raw CKA inflated by width
    for m in ("cka", "mknn"):
        assert abs(wide[m]["a"]["b"]["calibrated"]) < 0.05, wide[m]["a"]["b"]


def test_probe_recovers_planted_signal():
    rng = np.random.default_rng(0)
    m = 300
    X = rng.standard_normal((m, 40)).astype(np.float32)
    W = rng.standard_normal((40, 8))
    from scipy.spatial.transform import Rotation
    rv = rng.standard_normal((m, 3))
    X[:, 30:39] = Rotation.from_rotvec(rv).as_matrix().reshape(m, 9)   # eef_rot target lives in X linearly
    obj = X[:, :3] * 2 + 0.01 * rng.standard_normal((m, 3))
    state = (X @ W).astype(np.float32)
    state[:, 3:6] = rv
    L = {"state": state, "progress": X[:, 5], "obj": obj.astype(np.float32),
         "offset": (obj - state[:, :3]).astype(np.float32), "task": (X[:, 6] > 0).astype(np.int64),
         "group": np.arange(m) // 5}
    sc = analysis.probe_scores(X, L, n_comp=40, folds=3)
    for k in ("eef_pos", "eef_rot", "gripper", "progress", "obj_pos", "offset"):
        assert sc[k] > 0.9, (k, sc[k])
    assert sc["task"] > 0.9


def test_cluster_boot_and_holm():
    arr = np.zeros((30, 30))
    assert np.allclose(analysis.cluster_boot(arr, 100), 0)
    assert np.allclose(analysis.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_weight_diff_known_answers(tmp_path):
    from safetensors.torch import save_file
    g = torch.Generator().manual_seed(0)
    w = {"blocks.0.attn.weight": torch.randn(4, 4, generator=g), "patch_embedding.weight": torch.randn(3, 2, 1, 2, 2, generator=g)}
    save_file(w, tmp_path / "p.safetensors")
    child = {"transformer.blocks.0.attn.weight": w["blocks.0.attn.weight"] * 1.1,          # 10% change
             "transformer.patch_embedding_mlp.weight": w["patch_embedding.weight"].reshape(3, -1).clone(),  # reshaped, same
             "transformer.action_proj.weight": torch.zeros(2, 2)}                          # child-only
    save_file(child, tmp_path / "c.safetensors")
    r = analysis.weight_diff([str(tmp_path / "p.safetensors")], [str(tmp_path / "c.safetensors")], analysis.LADDER["lingbot_va"][2])
    assert r["rel"]["blocks.0"] == pytest.approx(0.1, rel=1e-5)
    assert r["rel"]["patch_embedding"] == 0.0
    assert r["child_only"] == ["transformer.action_proj.weight"] and not r["parent_only"]


def test_every_configured_model_has_an_adapter():
    c = run.CONFIG
    models = set(c["tier1"] + c["tier2"] + c["ablations"] + c["offline_only"]) - {"blind"}
    assert models <= set(encoders.ADAPTERS), models - set(encoders.ADAPTERS)
    assert set(c["prompt_invariant"]) <= set(c["text_models"]) and set(c["site_rule"]) <= models


def test_review_regressions(cfg, tmp_path):
    for m in ("blind", "stub"):
        for stage in ("calibrate", "extract", "train", "eval"):
            run.main(["--model", m, "--stage", stage, "--device", "cpu"])
    # token caches are deleted once a suite's heads exist; re-running extract/train is a no-op
    assert not list((tmp_path / "cache" / "tokens").rglob("*.npy"))
    run.main(["--model", "stub", "--stage", "extract", "--device", "cpu"])
    assert not list((tmp_path / "cache" / "tokens").rglob("*.npy"))
    # a head retune (G6) changes the head path and invalidates cached evals
    site = run.results_for(cfg, "stub").get("site", "name")
    retuned = {**cfg, "head": {**cfg["head"], "steps": 31}}
    assert policy.head_path(cfg, "stub", site, "libero_spatial", 0, "generic", True) != \
        policy.head_path(retuned, "stub", site, "libero_spatial", 0, "generic", True)
    # G6 reads both evals at the same proprio condition and head
    policy.gate_g6(cfg, margin=-1.0, pilot="stub")   # stub env: both 0.5
    with pytest.raises(AssertionError):
        policy.gate_g6(cfg, margin=0.1, pilot="stub")
    # cells are paired on exactly cfg seeds' init states
    assert analysis.cell_matrix(cfg, "stub").shape == (10, 4)
    assert analysis.cell_matrix({**cfg, "seeds": [0]}, "stub").shape == (10, 2)
    assert analysis.cell_matrix({**cfg, "seeds": [0, 1, 2]}, "stub") is None
    assert analysis.cell_matrix(cfg, "never_ran") is None and not (tmp_path / "results" / "never_ran.json").exists()
    # cut level persists across invocations
    run.main(["--stage", "analyze", "--cut-level", "2"])
    assert (tmp_path / "results" / "cut_level").read_text() == "2"
    assert "cut_level=2" in (tmp_path / "results" / "summary.md").read_text()


def test_xattn_read_splits_cross_attention():
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(2, 4, n, 6, generator=g) for n in (5, 9, 9))
    keep = torch.rand(2, 9, generator=g) > 0.3
    keep[:, 0] = True
    W = torch.randn(20, 24, generator=g)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=keep[:, None, None])
    ref = ref.transpose(1, 2).flatten(2) @ W.T
    r = encoders.xattn_read(q, k, v, torch.zeros(2, 9).masked_fill(~keep, float("-inf"))[:, None, None], W)
    assert torch.allclose(r.sum(1), ref.sum(1), atol=1e-4) and torch.all(r[~keep] == 0)
