# DINOv2 + LoRA + CLIP vs. R3M Benchmark

A reproducible benchmark pitting a parameter-efficient DINOv2 + LoRA + CLIP-text visual representation against R3M (Nair et al., CoRL 2022) on R3M's published behavior-cloning protocol over Franka Kitchen, MetaWorld, and Adroit.

**Hardware target:** 1× NVIDIA A100 80GB. **Wall clock:** ≤ 4 hours.

---

## TL;DR

```bash
# 1. Install deps
pip install "torch>=2.3" torchvision pillow numpy pandas pyarrow matplotlib \
            open_clip_torch nbformat decord       # decord: optional, used for Ego4D preprocessing
# heavyweight sim deps — only needed for actual BC rollouts
pip install git+https://github.com/Farama-Foundation/Metaworld.git
# install mj_envs from R3M's fork (provides the modified Franka Kitchen + Adroit envs)
git clone https://github.com/facebookresearch/r3m.git
cd r3m && pip install -e . && cd ..

# 2. Either:
#    (a) point CONFIG at already-preprocessed data:
#        CONFIG['ego4d_root'] = '/path/to/ego4d_r3m_frames'
#        CONFIG['demo_root']  = '/path/to/r3m_demos'
#    (b) preprocess raw Ego4D to the expected layout: enable RUN_PREPROCESSING=True
#        in cell P2 and point P2's EGO4D_VIDEOS_DIR / EGO4D_NARRATIONS_FILE at your
#        Ego4D download.

# 3. Open and run the notebook
jupyter lab dinov2_r3m_benchmark.ipynb
# Run all cells. Smoke tests gate the heavy production run automatically.
```

### Running on Google Colab

The notebook auto-detects Colab. Cells 5-6 mount Drive, clone the repo into `/content/`, check dependencies, and patch `CONFIG` paths so caches and the 2-hour pretrain checkpoint persist on Drive (not on Colab's ephemeral `/content/`). Just run cells in order — the Colab-specific paths are picked up automatically.

---

## What you get

After a successful run:

```
results/
├── results.json        # complete run record (config, every loss tick, every BC run + history)
├── side_by_side.png    # per-env bar chart, ours vs R3M
├── per_task.png        # per-task bars (Franka / MW / Adroit)
├── pretrain_curves.png # TCN + lang + total loss vs step
└── summary.md          # auto-generated markdown report — read this for the morning presentation
```

---

## Dependencies

### Python packages

| Package | Why |
|---|---|
| `torch >= 2.3` | bf16, `torch.compile`, current `torch.hub` API |
| `torchvision` | image transforms |
| `pillow` | JPEG load + resize (uses `Image.Resampling.BICUBIC`) |
| `numpy`, `pandas` | manifest handling, aggregation |
| `pyarrow` | reads/writes `manifest.parquet` |
| `matplotlib` | side-by-side bars, training curves |
| `open_clip_torch` | CLIP `ViT-B-32-quickgelu` (must be the `-quickgelu` variant for `pretrained='openai'`) |
| `decord` | optional — fast video decoder used by Ego4D preprocessing. Falls back to `opencv-python` if not installed. |
| `nbformat` | optional, only for re-validating the notebook |

### Sim dependencies (only needed for BC rollouts in the production run)

- **`metaworld`** — `pip install git+https://github.com/Farama-Foundation/Metaworld.git`
- **`mj_envs` (R3M's fork)** — provides the modified Franka Kitchen with desk randomization and the Adroit envs. Install from `facebookresearch/r3m`'s repo.

The smoke tests stub the simulator and **do not require either of these** — useful for first-pass verification on a fresh box before the heavyweight installs.

### Hardware

- 1× A100 80GB recommended. Smaller GPUs may need `CONFIG['batch_clips'] = 8` (halved from 16) — train for the same number of total samples by doubling pretrain steps, or accept fewer cross-video negatives per anchor.
- ~50 GB free disk: caches (CLIP text, pretrain checkpoint, per-(task, view) features) total ~30 GB; results dir ~5 MB.

---

## Required data

The notebook does **not** download Ego4D or generate the demo data. Ego4D access requires registration at https://ego4d-data.org. The demo `.pkl` files come from R3M's eval pipeline.

### 1. Ego4D pre-extracted frames

Expected layout under `CONFIG['ego4d_root']`:

```
ego4d_root/
├── manifest.parquet                # columns: clip_uid (str), narration (str), n_frames (int)
├── <clip_uid_001>/
│   ├── 0.jpg                       # I0 from first 20% of clip
│   ├── 1.jpg                       # Ii (middle frame 1)
│   ├── 2.jpg                       # Ij (middle frame 2)
│   ├── 3.jpg                       # Ik (middle frame 3)
│   └── 4.jpg                       # Ig from last 20% of clip
├── <clip_uid_002>/
│   ...
```

R3M's repo does **not** ship a preprocessing script — their README says the dataset is assumed to already be parsed into frames. Two options:

**Option A: you already have the frames cached** (e.g. from a prior R3M run). Point `CONFIG['ego4d_root']` at the directory and skip the preprocessing cells (`RUN_PREPROCESSING = False`, the default).

**Option B: preprocess raw Ego4D in the notebook.** Cells P0-P3 (just after Cell 4) handle this:
1. Download raw Ego4D via the [Ego4D CLI](https://github.com/facebookresearch/Ego4d) after registering: `ego4d --output_directory <data_root> --datasets full_scale annotations`. You get `<data_root>/v2/full_scale/<video_uid>.mp4` and `<data_root>/v2/annotations/narrations.json`.
2. Set `EGO4D_VIDEOS_DIR` and `EGO4D_NARRATIONS_FILE` in cell P2.
3. Set `RUN_PREPROCESSING = True` in cell P2.
4. Set `subset_n=50000` (or larger if you have more time/disk). Full Ego4D (~5M narrations) is 1-2 weeks of single-process work; 50k narrations on Colab CPU is ~40-90 min.
5. Run cells P0 (clone R3M for reference) → P1 (define preprocess function) → P2 (run) → P3 (validate).

Preprocessing is idempotent — re-running with `skip_existing=True` (default) picks up where it left off, so a Colab session disconnect mid-preprocessing only loses the current `log_every`-clip checkpoint window. Output is written incrementally; the manifest is checkpointed every 500 clips.

The preprocessor uses `decord` for fast random-access video decoding; if `decord` isn't installed (e.g. on arm64 Mac), it falls back to OpenCV per-frame seek (~3× slower but portable).

### 2. R3M demo episodes

Expected layout under `CONFIG['demo_root']`:

```
demo_root/
├── franka_kitchen/
│   ├── slide/view1/demo_0.pkl, demo_1.pkl, ...
│   ├── slide/view2/...
│   ├── light/view1/...
│   └── ...
├── metaworld/
│   ├── assembly-v2/view1/...
│   └── ...
└── adroit/
    ├── pen-v0/view1/...
    └── relocate-v0/view1/...
```

Each `.pkl` is a dict:
```python
{
    'images':  np.ndarray,  # shape (T, H, W, 3), uint8
    'actions': np.ndarray,  # shape (T, A), float32
    'proprio': np.ndarray,  # shape (T, P), float32 — joint pos/vel etc, env-specific
    'success': bool,        # whether the demo episode reached goal
}
```

These are produced by R3M's eval pipeline. Confirm camera names per view and proprio composition match what your `mj_envs` version exposes — the `# TODO: confirm` flags in Cell 15 surface as exceptions on the first BC call if there's a mismatch.

---

## How to run

### Step 1 — set paths

Two modes:

- **Local box**: open `dinov2_r3m_benchmark.ipynb`, edit **Cell 4 (CONFIG)** to point at your data:
  ```python
  CONFIG['ego4d_root'] = '/data/ego4d_r3m_frames'   # your Ego4D cache
  CONFIG['demo_root']  = '/data/r3m_demos'          # your demo .pkl tree
  ```
- **Google Colab**: just run cells in order. Cells 5-6 auto-mount Drive and patch `CONFIG` paths to `/content/drive/MyDrive/cs6998-final-proj-data/...`. Edit those defaults in cell 6 if your Drive layout differs.

Optionally set `CONFIG['cut_level']` (0=full, 1=drop Adroit, 2=demo sizes 3→2, 3=seeds 3→2, 4=drop view 3, 5=pretrain 150k→100k). Default is 1.

### Step 1.5 — preprocess Ego4D (only if needed)

Skip if you already have `<ego4d_root>/manifest.parquet` + `<clip_uid>/{0..4}.jpg`. Otherwise see the "Option B: preprocess raw Ego4D in the notebook" path under **Required data** above.

### Step 2 — run smoke tests first

The smoke section (Cells 21-23) builds synthetic micro-data, exercises every pipeline stage, and asserts each stage's contract. **Run this before kicking off the production run.**

- ~2 minutes on A100 (after one-time DINOv2 ~330 MB + CLIP ~600 MB downloads)
- ~5 minutes on CPU
- Stubs the simulator (real sim envs are exercised on the first BC sweep call)

Look for `=== ALL SMOKE TESTS PASSED ===` at the end. If you see an `AssertionError`, the failing assertion describes the contract that broke — usually a path/version issue surfaces here.

### Step 3 — run the production cells (Cells 25-30)

Once smoke passes, run cells in order. Each stage is independently re-runnable:

| Cell | Stage | Skip condition |
|---|---|---|
| 32 (PROD 1) | CLIP text cache (3-5 min) | `cache/clip_text_embeds.pt` exists |
| 33 (PROD 2) | Pretrain (~2.0-2.3 h) | `cache/pretrain.pt` exists |
| 34 (PROD 3) | Feature extraction (~10-15 min) | per-(task, view) cache hit |
| 35 (PROD 4) | BC sweep (~60-80 min) | always runs |
| 36 (PROD 5) | Aggregation + plots | always runs |
| 37 (PROD 6) | Markdown summary | always runs |

To force a re-run of any stage, delete its cache file. To skip the smoke tests on subsequent runs, set `RUN_SMOKE_TESTS = False` in Cell 21.

### Step 4 — read the report

Open `results/summary.md` and the three PNGs in `results/`. The summary has the per-environment delta vs R3M, per-task breakdown, and reproduction notes.

---

## Configuration knobs

Everything lives in `CONFIG` (Cell 4). Highlights:

| Knob | Default | Purpose |
|---|---|---|
| `pretrain_steps` | 150,000 | Pretrain budget. R3M used 1M; we cut by 6.7× citing DINOv2's strong init. |
| `batch_clips` | 16 | Video clips per batch (× 5 frames = 80 imgs/step). R3M's value. |
| `lora_rank`, `lora_alpha` | 16, 16 | LoRA capacity (~2.4M trainable params). |
| `lambda_tcn`, `lambda_lang`, `lambda_l1`, `lambda_l2` | 1, 1, 1e-5, 1e-5 | Loss weights from R3M Appendix A.2 (paper exact). |
| `bc_lr`, `bc_batch`, `bc_steps`, `bc_eval_every` | 1e-3, 32, 20k, 1k | BC protocol from R3M Appendix B.5 (paper exact). |
| `bc_eval_episodes` | 10 | Cut from R3M's 25 (saves ~2.5×; widens std-err to ±5 pp). |
| `bc_early_stop_patience` | 3 | Stop BC after 3 consecutive non-improving evals. |
| `cut_level` | 1 | Scope reduction ladder. See ARCHITECTURE.md §4.15. |
| `compile_model`, `precision` | True, 'bf16' | bf16 + `torch.compile(reduce-overhead)`. Disable compile if you see recompile warnings. |

---

## Output artifacts

```
cache/                              # disk-cached intermediate state (not committed)
├── clip_text_embeds.pt             # ~10 GB at full Ego4D
├── pretrain.pt                     # ~10 MB (LoRA + lang_head only)
└── features/<env>/<task>__<view>.pt # per-(task, view) demo features

results/                            # the deliverable
├── results.json                    # config + every pretrain loss tick + every BC run with full eval history
├── side_by_side.png
├── per_task.png
├── pretrain_curves.png
└── summary.md                      # auto-generated, the morning-presentation report
```

`results.json` is the single source of truth for any post-hoc analysis. The plots and markdown are derived from it.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `FileNotFoundError: manifest.parquet` | `CONFIG['ego4d_root']` wrong or no preprocessing run | re-check path; or set `RUN_PREPROCESSING=True` in cell P2 to preprocess in-notebook |
| `decord` install fails on arm64 Mac | no prebuilt wheel | the preprocessor falls back to OpenCV automatically; or skip preprocessing locally and run only on Linux/Colab |
| Preprocessing slow (>200 ms/clip) | OpenCV fallback being used | `pip install decord` for ~3× speedup on Linux/Colab |
| `KeyError` in MetaWorld env lookup | env name strings differ across MetaWorld versions | the cell tries both `ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE` and `ALL_V2_ENVIRONMENTS`; if both miss, check your installed metaworld version's keys |
| Black/zero rendered frames | wrong camera name for the view | edit `view_to_camera` in Cell 15; visual-check first frame |
| `ImportError: pyarrow` lib not loaded | broken pyarrow install | `pip install --force-reinstall pyarrow` |
| `UserWarning: QuickGELU mismatch` | using `'ViT-B-32'` instead of `'ViT-B-32-quickgelu'` | already fixed in CONFIG; if you see this you're not on the latest notebook |
| OOM during pretrain | shared GPU or other tenant | `CONFIG['batch_clips'] = 8` |
| `torch.compile` recompile warnings spam | LoRA shape changes per call | set `CONFIG['compile_model'] = False` |
| `_pickle.PicklingError: Can't pickle FeatureBCDataset` | spawn-based multiprocessing on Mac/Windows | already fixed (BC inner loader uses `num_workers=0`); if you re-introduce workers, define `FeatureBCDataset` in a real .py module |
| BC sweep skipped with `no pretrained model available` | pretrain didn't finish or checkpoint path is wrong | check Cell 26 output; confirm `cache/pretrain.pt` exists |
| Smoke tests pass but production hangs at first rollout | sim env env-name or proprio mismatch | the rollout will eventually time out or crash — check the `# TODO: confirm` flags in Cell 15 against your `mj_envs` install |

---

## Re-running individual stages

```bash
# re-run pretrain only (delete the checkpoint)
rm cache/pretrain.pt && jupyter nbconvert --execute --to notebook --inplace dinov2_r3m_benchmark.ipynb

# re-run feature extraction only (delete cached features)
rm -rf cache/features/

# re-run BC + aggregation only (don't delete anything)
# Just run Cells 28-30 manually
```

---

## Files in this project

| File | Purpose |
|---|---|
| `dinov2_r3m_benchmark.ipynb` | The deliverable — 37-cell runnable benchmark |
| `ARCHITECTURE.md` | Full design document — motivation, alternatives considered per component, pros/cons, risk register |
| `README.md` | This file — operational instructions |
| `build_notebook.py` | Builds the notebook from a single Python source (easier to diff than raw .ipynb JSON). Run with `python3 build_notebook.py` to regenerate. |
| `smoke_test.py` | Standalone CPU smoke test against synthetic data for the core pipeline. Independent of the notebook. |
| `smoke_test_preprocess.py` | Standalone smoke test for `preprocess_ego4d` against synthetic mp4 + narrations.json. |
| `run_notebook_offline.py` | Integration test that exec()s notebook cells in order with DINOv2 / pyarrow / sim deps stubbed. Used to verify the smoke section integrates cleanly with the cell definitions. |
| `2203.12601v3.pdf` | The R3M paper (reference) |
| `paper.txt` | Plain-text extract of the paper (used during development to look up exact numbers) |

---

## References

- **R3M paper:** Nair et al., *R3M: A Universal Visual Representation for Robot Manipulation*, CoRL 2022. arXiv:[2203.12601](https://arxiv.org/abs/2203.12601). Local: `2203.12601v3.pdf`.
- **R3M code & eval pipeline:** [facebookresearch/r3m](https://github.com/facebookresearch/r3m).
- **DINOv2:** Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2023. [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2). Loaded via `torch.hub`.
- **CLIP / open_clip:** Radford et al., 2021. We use `'ViT-B-32-quickgelu'` with `pretrained='openai'`.

For the rationale behind every architectural choice, see **`ARCHITECTURE.md`**.
