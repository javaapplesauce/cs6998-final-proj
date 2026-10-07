# Architecture — v2 representation study

**Question.** How do 2026 foundation models (frozen perception encoders, VLA internals, generative world
models, embodied VLM reasoners) represent the manipulation world, and what does that do to low-level control?
Plan of record: `~/.claude/plans/this-project-is-based-vivid-squirrel.md` (hypotheses H1–H5, §1.1).
v1 (DINOv2/DINOv3/SAM 2 vs R3M notebooks) lives in `legacy/` and never ran on real data.

## Pipeline

```
lerobot/libero @ a1aaacb7  ──decode once──▶  cache/frames/<suite>.npy (uint8 256², dataset orientation)
                                                   │
             ┌─────────── encoders.load(model) ────┤ 64 tokens/frame, frozen, generic prompt
             ▼                                     ▼
  calibrate: 4 depth-matched sites ×      extract: selected site, all 171,996 frames
  10% of episodes → brief head → held-out        │  cache/tokens/<model>/<site>__<text>/<suite>.npy (fp16)
  chunk L1 → site (or ladder-matched site)       ▼
                                          train: 1 head per (suite, seed), fixed steps, final ckpt
                                                 │  results/heads/<model>/<site>/...pt
                                                 ▼
                                          eval: LIBERO closed loop, 30 sub-envs/task,
                                                seed s ↔ init states 10s…10s+9, chunks open-loop
probe set (6k dataset + 8k replayed frames,       │
object positions logged) ──▶ probe: all sites ──┐ │
                                                ▼ ▼
                         analyze: probes, calibrated CKA / mutual-kNN, contrasts, link → summary.md
```

One process per `(model, stage)`: `python -m repstudy.run --model M --stage S`. Every number in
`results/summary.md` traces to `results/<model>.json` (schema: `model, config, meta, gates, calibrate, site,
train, eval, probe`; `run.Results` rejects any other section).

## Modules

| file | owns |
|---|---|
| `repstudy/run.py` | `CONFIG`, `apply_cuts` (§1.9 cut ladder), `Results` (atomic per-model JSON), proprio rule, site rule, stage CLI |
| `repstudy/encoders.py` | adapter contract (`Encoder(name, sites, fn, text)`), helpers (`to_tensor`, `pool8`, `capture`), registry, stub |
| `repstudy/extract.py` | frame cache (lerobot's own `decode_video_frames`, so frames match `LeRobotDataset`), token cache + per-dim stats, site sweep, probe-set builder, gates G1/G2/G5 |
| `repstudy/policy.py` | `Head`, chunk dataset, `fit`, LIBERO env construction/IO, `rollout`/`evaluate`, demo replay, gates G0/G3/G7 |
| `repstudy/analysis.py` | probes, null-calibrated similarity, cluster bootstrap + Holm, Spearman link, weight-diff audit, tables + figures |
| `tests/test_smoke.py` | G4: CPU end to end with stub encoder + stub vector env, no lerobot import |
| `run_colab.ipynb` | thin driver over the CLI |

## Protocol (§1.4) as implemented

- **Inputs.** Agentview only, 256², dataset orientation. Env frames pass through lerobot's
  `preprocess_observation` + `LiberoProcessorStep` (180° rotation; 8-d state = eef pos, eef axis-angle,
  gripper qpos). Proprio (8-d) and the 10-way task one-hot reach the head only.
- **Text.** Primary condition: `GENERIC_PROMPT = "What should the robot do next?"` for every text model, so no
  model sees the task in words. `--text instruction` is the labelled secondary condition (offline site curve
  for all text models; closed loop for the L3 pair while `closed_loop_secondary`).
- **Tokens.** Exactly 64 per frame (8×8, row-major); dense grids are average-pooled by `pool8`. Tokens are
  standardised per dimension with statistics over the suite's cache; the statistics live in the head's buffers.
- **Readout site.** Each adapter exposes four depth-matched sites (eight for the video models: block ×
  {observed, predicted}). `calibrate` encodes ~10% of episodes per suite at every site, trains the real head
  for `calib.steps` on 80% of those episodes and scores held-out chunk L1. Site = argmin, except ladder rungs:
  `site_rule` fixes the L1 ladder at LLM layer 16 and makes Wan2.2 use LingBot-VA's chosen site index.
- **Head.** `Linear(D→384)+LayerNorm → 1 learned-query MHA pool (8 heads) → concat(proprio, one-hot) →
  MLP(512, 512) → 8×7 chunk`, L1 loss, AdamW 3e-4, cosine, batch 256, 20k steps, final checkpoint only.
  Identical for every model; tuned once at the G6 pilot, then frozen.
- **Evaluation.** One AsyncVectorEnv per task with one sub-env per initial state (`LiberoEnv(episode_index=
  init)`), so seed s always runs init states 10s…10s+9 for every model. Three seeds' heads act on their own
  slices of the same batch; the encoder is called once per 8-step chunk. Success latches on
  `info["is_success"]`; the step cap (280/280/300) is enforced by `rollout`. `FreezeAfterEpisodeEnd` makes
  finished sub-envs free.

## Controls (§1.5)

- **Blind**: model `blind` = head with proprio + one-hot, no tokens. Rule fixed in advance
  (`run.proprio_on`): if blind exceeds 30% on a suite, every model (blind included) drops proprio there; the
  rule always reads blind's with-proprio result (`<suite>|generic|p1`), so it cannot flip-flop.
- **Random floor**: `random_vit` = V-JEPA 2.1 architecture, random init, seed 0.
- Token-budget (16×16) and Wan 512² resolution ablations are adapter options, not separate code paths.

## Representation analysis (§1.6)

- **Probe set** (`cache/probe/`): 6k dataset frames (eef pose, gripper, progress, task) + ~8k frames from
  replaying demo actions in a single in-process `LiberoEnv` from the init state nearest the demo's first frame;
  these also carry the target-object position (the object that moves most) and the gripper→object offset.
  CV groups = dataset episode id for both sources, so a demo and its replay never straddle folds.
- **Probes**: per site, 2×2-pooled tokens → fold-wise standardise + PCA(512) → RidgeCV with episode-grouped
  inner alpha selection (R², mean over outputs) / logistic regression (task accuracy), 5 episode-grouped folds. eef rotation is probed as a rotation
  matrix (axis-angle wraps at π).
- **Similarity**: mean-pooled tokens on a fixed 4k-frame subset; linear CKA (centred Grams) and mutual k-NN
  (k=10). Null = row permutation of one side (100×); calibrated = (raw − null mean)/(1 − null mean), plus the
  permutation p-value. Hierarchical clustering + MDS on 1 − calibrated CKA.
- **Link**: Spearman across models between probe scores at the selected site and mean success (exploratory).

## Statistics (§1.7)

Cells are (suite, task, init state) — identical across models by construction; `cell_matrix` keeps exactly the
configured seeds' init states and refuses to pair models evaluated on different sets. `cluster_boot` resamples
(suite, task) clusters keeping each cluster's mean over init states (resampling inits again would count the
within-cluster variance twice); contrasts are paired cell differences; two-sided bootstrap p floored at
1/(n_boot+1), Holm over the four planned contrasts (a group contrast run on a subset says so in its label).
Per-suite CIs use the same bootstrap; seed spread is reported as SD.

## Gates (§3)

| gate | command | checks |
|---|---|---|
| G0 | `--stage g0` | `import libero` in a subprocess with stdin closed, step caps and `NEW_ROLLOUT_OPTION` match lerobot, one EGL frame |
| G1 | `--stage g1` | pinned revision, v3.0 keys/shapes, 30/30 task strings, contiguous suite episode ranges, 171,996 frames |
| G2 | `--stage g2` | nearest init state's first frame ≪ error of the un-rotated frame; eef state within 0.1, gripper within 0.01 |
| G3 | `--stage g3` | replayed demo actions succeed at least once (else stock `lerobot-eval` GR00T check) |
| G4 | `pytest tests/test_smoke.py` | end to end < 60 s; v1 bug classes; analysis known answers |
| G5 | `--model M --stage g5` | shapes, finiteness, determinism, rotation sensitivity (centred per-position cosine), prompt invariance for `CONFIG['prompt_invariant']`, dataset-vs-env cosine, fps, VRAM |
| G6 | `--stage g6` after the vjepa2_1 pilot (libero_spatial, seed 0) | ≥ blind + 0.10 at the same proprio condition and head hash, not at floor |
| G7 | `--stage g7` | env-steps/s + measured full-model stage times (plan §1.9 figures otherwise) → projected hours per cut level → smallest level within `REPSTUDY_BUDGET_H` |

## Adapters

See the table in `README.md` (sites, input resolution, native grid, D) and each `_<name>` docstring in
`repstudy/encoders.py`. Shapes there come from configs/source; G5 asserts them at runtime.
