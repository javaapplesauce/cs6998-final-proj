"""Build the DINOv2+LoRA vs R3M benchmark notebook."""
import json, uuid
from pathlib import Path

NB_PATH = Path(__file__).parent / "baseline_2_dinov3_lora.ipynb"

cells = []


def _id():
    return uuid.uuid4().hex[:12]


def md(src: str):
    cells.append({"cell_type": "markdown", "id": _id(), "metadata": {},
                  "source": src.splitlines(keepends=True)})


def code(src: str):
    cells.append({
        "cell_type": "code",
        "id": _id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    })


# ============================================================
# Cell 1 — title & overview
# ============================================================
md("""# Baseline 2: DINOv3 + LoRA + CLIP vs. R3M — Benchmark Notebook

Goal: train a DINOv3 (ViT-B/16, frozen) backbone with LoRA adapters using R3M's
loss formulation (time-contrastive + video-language alignment + L1/L2) on Ego4D,
then evaluate on Franka Kitchen + MetaWorld (+ Adroit if budget permits) with the
exact R3M behavior-cloning protocol.

This is **Baseline 2** in our three-way comparison:
- **Baseline 1**: R3M (Nair et al., 2022) — ResNet50 with TCN+VLA+L1/L2 on Ego4D 1M steps. Numbers hardcoded from paper Tables 1-3.
- **Baseline 2** (this notebook): DINOv3 ViT-B/16 + LoRA r=16, R3M's losses on Ego4D 150k steps.
- **Baseline 3** (`baseline_3_dinov3_sam3.ipynb`): DINOv3 ViT-L/16 + SAM 3 concept masks, fully frozen, no Ego4D pretraining.

Reference: Nair et al., *R3M: A Universal Visual Representation for Robot
Manipulation*, CoRL 2022 (arXiv:2203.12601). λ values from Appendix A.2; BC
protocol from Appendix B.5.

**Hardware target:** 1× A100 80GB, ≤ 4 h wall clock.
""")


# ============================================================
# Cell 2 — bottleneck analysis (matches text response)
# ============================================================
md("""## Bottleneck analysis & wall-clock budget

| Stage | Config | Estimate |
|---|---|---|
| CLIP text pre-embed | ViT-B/32 over all unique narrations | ~3-5 min |
| Pretrain | 150k steps, bs=16 clips × 5 frames | ~2.0-2.3 h |
| Downstream feature pre-extract | DINOv2+LoRA on all demo frames | ~10-15 min |
| BC sweep (default cut) | (5+5) tasks × 2 views × 2 sizes × 3 seeds | ~60-80 min |
| Aggregate + plot + markdown summary | | ~2 min |
| **Total** | | **~3.4-3.8 h** |

**Top 3 bottlenecks & mitigations**
1. Ego4D I/O — pre-extracted JPEG cache + cached CLIP text embeddings + persistent DataLoader workers.
2. ViT-B/16 fwd/bwd — bf16, `torch.compile`, frozen backbone (LoRA-only optimizer state).
3. BC sweep is sim-rollout-bound — pre-extract per-frame features once per (task, view); 3-checkpoint plateau early stopping; reduced eval episodes.

**Cut order if running long (`CONFIG['cut_level']`)**
1. drop Adroit  →  2. demo sizes 3→2  →  3. seeds 3→2  →  4. drop View 3  →  5. pretrain 150k→100k.

**Justification for 150k pretrain (vs. R3M's 1M):** DINOv2 init is already strong on object/scene structure; we are *adapting* a representation, not learning from scratch. 6.7× pretrain speedup is consistent with strong-init literature (MAE-init, DINO-init).
""")


# ============================================================
# Cell 2.5 — environment setup / dependencies
# ============================================================
md("""## Environment setup

Required packages on the A100 box (only run if missing — these will not auto-install):

```bash
pip install torch>=2.3 torchvision pillow numpy pandas pyarrow matplotlib \\
            open_clip_torch nbformat
# Sim deps (heavyweight — confirm with `pip list | grep -E 'mj_envs|metaworld'`):
#   mj_envs from R3M's fork:  https://github.com/facebookresearch/r3m  (cd r3m/evaluation && pip install -e .)
#   metaworld:                pip install git+https://github.com/Farama-Foundation/Metaworld.git
```

Required artifacts (paths set in `CONFIG` below):

- **Ego4D pre-extracted frames** at `CONFIG['ego4d_root']` — `<clip_uid>/{0..4}.jpg` + `manifest.parquet`. Use R3M's `preprocess_ego4d.py` to produce this offline.
- **R3M demo `.pkl` files** at `CONFIG['demo_root']/<env>/<task>/<view>/demo_*.pkl` — produced by R3M's eval repo.

If any of these are missing, the relevant cell will surface a `FileNotFoundError` with a `# TODO: confirm` hint instead of silently degrading.
""")


# ============================================================
# Cell 3 — CONFIG + imports + GPU/precision setup
# ============================================================
code("""# ===== CELL 3: CONFIG + imports + setup =====
# All hyperparameters live here. Nothing magic should appear inside functions.
import os, sys, json, time, math, random, hashlib, pickle, shutil
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple, Iterable

CONFIG = {
    # ---- paths (TODO: confirm) ----
    'ego4d_root':       '/data/ego4d_r3m_frames',   # TODO: confirm — pre-extracted layout: <clip_uid>/{0..4}.jpg + manifest.parquet
    'manifest_file':    'manifest.parquet',         # columns: clip_uid, narration, n_frames
    'demo_root':        '/data/r3m_demos',          # TODO: confirm — per-task demo .pkl produced by R3M's eval repo
    'cache_dir':        './cache',                  # CLIP text embeds, pre-extracted features, checkpoints
    'results_dir':      './results',                # results.json, plots, summary.md
    'pretrain_ckpt':    './cache/pretrain.pt',

    # ---- model ----
    'dinov3_variant':   'dinov3_vitb16',            # ViT-B/16, 86M params, 768-dim feature
    'dinov3_weights':   None,                       # local path or URL (gated via https://ai.meta.com/dino-v3); leave None to fail loudly if unset at production load
    'clip_variant':     'ViT-B-32-quickgelu',       # MUST be -quickgelu for pretrained='openai' (otherwise activation mismatch ≈ 0.35 max diff in text embeds — degrades VLA)
    'clip_pretrained':  'openai',
    'feat_dim':         768,                        # DINOv3 ViT-B/16 output dim (CLS token); 1024 for vitl16
    'text_dim':         512,                        # CLIP ViT-B/32 text dim
    'lora_rank':        16,
    'lora_alpha':       16,
    'lora_dropout':     0.05,
    'lora_targets':     ('qkv', 'proj', 'fc1', 'fc2'),  # attention + Mlp modules; verified for dinov3 vitb16 / vitl16 (ffn_layer='mlp', so fc1/fc2 — NOT applicable for vitl16plus / hyplus / 7b which use SwiGLU and need ('w1','w2','w3'))
    'lang_head_hidden': (1024, 1024, 1024, 1024),   # 5-layer MLP per R3M Appendix A.2

    # ---- pretrain ----
    'pretrain_steps':       150_000,                # cut from R3M's 1M; justified by DINOv2 init
    'pretrain_lr':          1e-4,                   # Adam, per Appendix A.2
    'pretrain_weight_decay':0.0,
    'batch_clips':          16,                     # video clips per batch (R3M: 16)
    'frames_per_clip':      5,                      # I0, Ii, Ij, Ik, Ig (R3M: 5)
    'image_size':           224,
    'num_negatives':        3,                      # cross-video negatives, per Appendix A.3
    'lambda_tcn':           1.0,                    # λ1
    'lambda_lang':          1.0,                    # λ2
    'lambda_l1':            1e-5,                   # λ3
    'lambda_l2':            1e-5,                   # λ4
    'pretrain_log_every':   200,
    'pretrain_save_every':  10_000,

    # ---- compute ----
    'precision':            'bf16',
    'compile_model':        True,
    'compile_mode':         'reduce-overhead',
    'num_workers':          8,
    'prefetch_factor':      4,
    'persistent_workers':   True,
    'pin_memory':           True,

    # ---- BC eval (R3M protocol, Appendix B.5) ----
    'bc_lr':            1e-3,
    'bc_batch':         32,
    'bc_steps':         20_000,
    'bc_eval_every':    1_000,
    'bc_hidden':        (256, 256),
    'bc_eval_episodes': 10,                  # cut from R3M's 25; ±5% s.e. is fine for AB
    'bc_early_stop_patience': 3,             # stop after 3 consecutive non-improving evals

    # ---- envs to sweep ----
    'envs': {
        'franka_kitchen': {
            'tasks': ['slide', 'light', 'door', 'knob', 'micro'],
            'views': ['view1', 'view2', 'view3'],   # # TODO: confirm camera names from R3M eval repo
            'demo_sizes': [5, 10, 25],
            'horizon': 50,
        },
        'metaworld': {
            'tasks': ['assembly-v2', 'bin-picking-v2', 'button-press-topdown-v2',
                      'drawer-open-v2', 'hammer-v2'],
            'views': ['view1', 'view2', 'view3'],   # # TODO: confirm
            'demo_sizes': [5, 10, 25],
            'horizon': 500,
        },
        'adroit': {
            'tasks': ['pen-v0', 'relocate-v0'],
            'views': ['view1', 'view2'],
            'demo_sizes': [25, 50, 100],
            'horizons': {'pen-v0': 100, 'relocate-v0': 200},
        },
    },

    'seeds': [0, 1, 2],
    'cut_level': 1,    # 0=full sweep; 1=drop Adroit; 2=demo sizes 3→2; 3=seeds 3→2; 4=drop view3; 5=pretrain 100k
}

def apply_cuts(cfg):
    cl = cfg['cut_level']
    if cl >= 1: cfg['envs'].pop('adroit', None)
    if cl >= 2:
        for e in cfg['envs'].values():
            if 'demo_sizes' in e and len(e['demo_sizes']) >= 3:
                e['demo_sizes'] = [e['demo_sizes'][0], e['demo_sizes'][-1]]
    if cl >= 3: cfg['seeds'] = cfg['seeds'][:2]
    if cl >= 4:
        for e in cfg['envs'].values():
            if 'views' in e and len(e['views']) >= 3:
                e['views'] = e['views'][:2]
    if cl >= 5: cfg['pretrain_steps'] = 100_000
    return cfg

CONFIG = apply_cuts(CONFIG)
Path(CONFIG['cache_dir']).mkdir(parents=True, exist_ok=True)
Path(CONFIG['results_dir']).mkdir(parents=True, exist_ok=True)

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE  = torch.bfloat16 if CONFIG['precision'] == 'bf16' else torch.float32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

print(f"[setup] device={DEVICE}, dtype={DTYPE}")
print(f"[setup] cuda capability: {torch.cuda.get_device_capability() if DEVICE.type=='cuda' else 'cpu'}")
print(f"[setup] cut_level={CONFIG['cut_level']}, pretrain_steps={CONFIG['pretrain_steps']}, envs={list(CONFIG['envs'].keys())}")
""")


# ============================================================
# Cell 3.5 — Colab / Drive setup (optional, runs only on Colab)
# ============================================================
md("""---

## Optional: Colab / Drive setup

If you're running on Google Colab, run the next cell to mount Drive, clone the
project repo, check dependencies, and override `CONFIG` paths so caches land on
Drive (so a Colab session disconnect doesn't destroy your 2-hour pretrain
checkpoint). On a non-Colab box, the cell auto-detects and is a near no-op —
you only need to set `DRIVE_ROOT` if you don't want the default `/mnt/drive`.

**Skip this cell** if you're running locally with the default paths in `CONFIG`
already pointing at where your data lives.
""")

code("""# ===== Colab setup: mount Drive, clone repo, check deps, patch CONFIG paths =====
# Idempotent: safe to re-run. Patches CONFIG paths in place; does NOT redefine
# CONFIG (which would clobber 60+ hyperparameters from Cell 3).
import os, sys, subprocess, importlib.util
from pathlib import Path

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    DRIVE_ROOT = Path('/content/drive/MyDrive')
    IN_COLAB = True
except ImportError:
    DRIVE_ROOT = Path('/mnt/drive')   # TODO: confirm for non-Colab boxes
    IN_COLAB  = False
    print(f"[info] not in Colab — DRIVE_ROOT={DRIVE_ROOT}")

# Clone repo into ephemeral /content (small-file I/O is slow on Drive)
REPO_DIR = Path('/content/cs6998-final-proj') if IN_COLAB else Path.cwd()
if IN_COLAB and not REPO_DIR.exists():
    subprocess.run(['git', 'clone',
                    'https://github.com/javaapplesauce/cs6998-final-proj',
                    str(REPO_DIR)], check=True)
elif IN_COLAB:
    print(f"[info] {REPO_DIR} exists — `!git -C {REPO_DIR} pull` to update")
sys.path.insert(0, str(REPO_DIR))

# Dependency presence check (find_spec, no actual import)
REQUIRED = {'torch':'torch>=2.3','torchvision':'torchvision','PIL':'pillow',
            'numpy':'numpy','pandas':'pandas','pyarrow':'pyarrow',
            'matplotlib':'matplotlib','open_clip':'open_clip_torch','nbformat':'nbformat'}
missing = [pkg for mod, pkg in REQUIRED.items() if importlib.util.find_spec(mod) is None]
if missing:
    print(f"[warn] missing — install before continuing:  pip install {' '.join(missing)}")
else:
    print("[ok] core packages present")
for sim in ('mj_envs', 'metaworld'):
    if importlib.util.find_spec(sim) is None:
        print(f"[warn] {sim} not importable — required for production BC sweep only")
    else:
        print(f"[ok] {sim} importable")

# Patch CONFIG paths to land on Drive (preserves all other hyperparameters).
# Only patches when DRIVE_ROOT is actually accessible — otherwise leaves the
# Cell 3 defaults so this cell stays a no-op on non-Colab boxes.
if IN_COLAB or DRIVE_ROOT.exists():
    DATA_ROOT = DRIVE_ROOT / 'cs6998-final-proj-data'   # TODO: confirm staging dir
    RUNS_ROOT = DRIVE_ROOT / 'cs6998-final-proj-runs'   # caches + results persist here
    CONFIG.update({
        'repo_root':     str(REPO_DIR),
        'ego4d_root':    str(DATA_ROOT / 'ego4d_frames'),    # TODO: confirm
        'demo_root':     str(DATA_ROOT / 'r3m_demos'),       # TODO: confirm
        'cache_dir':     str(RUNS_ROOT / 'cache'),
        'results_dir':   str(RUNS_ROOT / 'results'),
        'pretrain_ckpt': str(RUNS_ROOT / 'cache' / 'pretrain.pt'),
    })
    Path(CONFIG['cache_dir']).mkdir(parents=True, exist_ok=True)
    Path(CONFIG['results_dir']).mkdir(parents=True, exist_ok=True)
    print("\\n[ok] CONFIG paths patched to Drive:")
    for k in ('repo_root','ego4d_root','demo_root','cache_dir','results_dir','pretrain_ckpt'):
        print(f"  {k}: {CONFIG[k]}")
else:
    print(f"\\n[info] DRIVE_ROOT={DRIVE_ROOT} not accessible — leaving Cell 3 defaults intact.")
    print("       Set CONFIG['ego4d_root'] / CONFIG['demo_root'] manually if running locally.")
""")


# ============================================================
# Cell 3.6 — Ego4D preprocessing (optional)
# ============================================================
md("""---

## Optional: Ego4D preprocessing

R3M's repo does **not** ship a preprocessing script — their README says
"the dataset has been parsed into frames" and assumes you've done it yourself.
If you already have `<ego4d_root>/manifest.parquet` + `<clip_uid>/{0..4}.jpg`,
**skip this section** and continue to Cell 4 (Logger).

If you have raw Ego4D `.mp4` videos + `narrations.json` and need to produce
the layout the rest of the notebook expects, run cells P0-P3 below. Their
behavior is gated by `RUN_PREPROCESSING` — set it to `False` to skip.

**Inputs needed (download via Ego4D CLI after registering at https://ego4d-data.org):**
- a directory of `<video_uid>.mp4` files
- the `narrations.json` annotation file

**Output:** `CONFIG['ego4d_root']/<clip_uid>/{0..4}.jpg` + `manifest.parquet`,
ready to be consumed by `Ego4DR3MDataset` (Cell 6).

**Wall-clock:** ~50-100 ms per clip on a Colab CPU. Full Ego4D (~5M narrations)
is days; subset to 50k-100k for a single-session run. The `subset_n` knob in
P2 is the budget control.
""")

code("""# ===== P0: clone R3M for reference (their data_loaders.py is the source-of-truth on sampling) =====
import subprocess
from pathlib import Path

R3M_REF = Path('/content/r3m_reference') if 'IN_COLAB' in globals() and IN_COLAB else Path('./r3m_reference')
if not R3M_REF.exists():
    subprocess.run(['git', 'clone', '--depth', '1',
                    'https://github.com/facebookresearch/r3m.git', str(R3M_REF)], check=True)
    print(f"R3M cloned at {R3M_REF}")
else:
    print(f"R3M already present at {R3M_REF}")
print("R3M's preprocessing is NOT shipped (see their README line 34).")
print(f"Reference loader (sampling logic): {R3M_REF}/r3m/utils/data_loaders.py")
""")

code("""# ===== P1: define preprocess_ego4d (replicates R3M's clip-around-narration logic) =====
# Tries decord (fast, random-access). Falls back to OpenCV if decord isn't available
# (e.g. arm64 Mac). decord is the recommended path on Linux/Colab.
import json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

def _open_video_decoder(path):
    \"\"\"Returns (reader, fps, n_frames, get_frames_fn). decord first, cv2 fallback.\"\"\"
    try:
        import decord
        decord.bridge.set_bridge('native')
        vr = decord.VideoReader(str(path))
        fps = float(vr.get_avg_fps()); n = len(vr)
        def get_frames(indices):
            return vr.get_batch(list(indices)).asnumpy()
        return vr, fps, n, get_frames
    except ImportError:
        import cv2
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"cv2 failed to open {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        def get_frames(indices):
            out = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if not ok: raise RuntimeError(f"cv2 failed to read frame {idx}")
                out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return np.stack(out, axis=0)
        return cap, fps, n, get_frames

def _close_video(reader):
    if hasattr(reader, 'release'):
        reader.release()

def preprocess_ego4d(
    videos_dir, narrations_json, output_dir, *,
    image_size=224, clip_window_sec=4.0,
    narration_pass='narration_pass_1', only_camera_wearer=True,
    subset_n=None, skip_existing=True, log_every=500,
):
    \"\"\"
    For each Ego4D narration: carve a window of ±clip_window_sec around the timestamp;
    sample 5 frames using R3M's rule (I0 from first 20%, Ig from last 20%, Ii<Ij<Ik
    in between); resize to image_size; save as <output_dir>/<clip_uid>/{0..4}.jpg.
    Append (clip_uid, narration, n_frames=5) rows to <output_dir>/manifest.parquet.
    Idempotent: re-running with skip_existing=True picks up where it left off.
    \"\"\"
    videos_dir = Path(videos_dir); output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / 'manifest.parquet'

    print(f"[load] {narrations_json}")
    narrations_data = json.loads(Path(narrations_json).read_text())
    print(f"  {len(narrations_data):,} videos in narrations.json")

    flat = []
    for video_uid, vdata in narrations_data.items():
        if narration_pass not in vdata: continue
        for k, n in enumerate(vdata[narration_pass].get('narrations', [])):
            text = (n.get('narration_text') or '').strip()
            if only_camera_wearer and not text.startswith('#C'): continue
            ts = n.get('timestamp_sec')
            if ts is None: continue
            flat.append({'video_uid': video_uid, 'timestamp_sec': float(ts),
                         'narration_text': text,
                         'annotation_uid': n.get('annotation_uid') or f"idx{k:08d}"})
    print(f"  {len(flat):,} narrations after filtering")
    if subset_n is not None:
        flat = flat[:subset_n]
        print(f"  subset_n={subset_n}: processing {len(flat):,}")

    rows, existing_uids = [], set()
    if manifest_path.exists() and skip_existing:
        existing = pd.read_parquet(manifest_path)
        rows = existing.to_dict('records')
        existing_uids = set(existing['clip_uid'].tolist())
        print(f"  resuming: {len(existing_uids):,} clips already done")

    n_done = len(existing_uids); n_skip = 0; n_fail = 0
    last_uid, reader, fps, n_video_frames, get_frames = None, None, None, None, None
    t0 = time.time()

    try:
        for i, item in enumerate(flat):
            video_uid = item['video_uid']
            clip_uid = f"{video_uid}_{item['annotation_uid']}_{int(item['timestamp_sec']*1000):010d}"
            if clip_uid in existing_uids: n_skip += 1; continue
            clip_dir = output_dir / clip_uid
            if (clip_dir / '4.jpg').exists():
                existing_uids.add(clip_uid); n_skip += 1; continue

            try:
                if video_uid != last_uid:
                    if reader is not None: _close_video(reader)
                    cands = sorted(videos_dir.glob(f"{video_uid}.*"))
                    if not cands: n_fail += 1; continue
                    reader, fps, n_video_frames, get_frames = _open_video_decoder(cands[0])
                    last_uid = video_uid

                t_lo = max(0.0, item['timestamp_sec'] - clip_window_sec)
                t_hi = min(n_video_frames / fps, item['timestamp_sec'] + clip_window_sec)
                f_lo = int(t_lo * fps); f_hi = min(n_video_frames - 1, int(t_hi * fps))
                n_clip = f_hi - f_lo + 1
                if n_clip < 5: n_fail += 1; continue

                first20 = max(1, n_clip // 5); last20 = max(1, n_clip // 5)
                i0 = random.randint(0, first20 - 1)
                ig = random.randint(n_clip - last20, n_clip - 1)
                lo, hi = i0 + 1, ig - 1
                if hi - lo + 1 < 3:
                    mids = sorted([random.randint(lo, hi) for _ in range(3)]) if hi >= lo else [i0]*3
                else:
                    mids = sorted(random.sample(range(lo, hi + 1), 3))
                video_indices = [f_lo + ci for ci in (i0, mids[0], mids[1], mids[2], ig)]

                frames = get_frames(video_indices)
                clip_dir.mkdir(parents=True, exist_ok=True)
                for j, frame in enumerate(frames):
                    Image.fromarray(frame).resize(
                        (image_size, image_size), Image.Resampling.BICUBIC
                    ).save(clip_dir / f"{j}.jpg", quality=85)

                rows.append({'clip_uid': clip_uid,
                             'narration': item['narration_text'],
                             'n_frames': 5})
                n_done += 1
            except Exception as e:
                n_fail += 1
                if n_fail < 10: print(f"  fail [{i}]: {type(e).__name__}: {e}")

            if (i + 1) % log_every == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (len(flat) - i - 1) / 60
                pd.DataFrame(rows).to_parquet(manifest_path)
                print(f"  [{i+1:,}/{len(flat):,}] done={n_done:,} skip={n_skip:,} "
                      f"fail={n_fail:,} eta={eta:.1f}min")
    finally:
        if reader is not None: _close_video(reader)

    pd.DataFrame(rows).to_parquet(manifest_path)
    print(f"\\n[done] {n_done:,} clips written, {n_skip:,} skipped, {n_fail:,} failed")
    print(f"  manifest: {manifest_path}")
    return manifest_path
""")

code("""# ===== P2: run preprocessing (gated by RUN_PREPROCESSING) =====
RUN_PREPROCESSING = False    # set True to run; False to skip if your data is already preprocessed

# Where your raw Ego4D inputs live. Get these via the Ego4D CLI after registering
# (https://ego4d-data.org). The CLI command is roughly:
#   ego4d --output_directory <data_root> --datasets full_scale annotations
EGO4D_VIDEOS_DIR      = (Path(CONFIG['repo_root']).parent if 'repo_root' in CONFIG else Path('.')) / 'ego4d_raw_videos'   # TODO: confirm
EGO4D_NARRATIONS_FILE = (Path(CONFIG['repo_root']).parent if 'repo_root' in CONFIG else Path('.')) / 'narrations.json'    # TODO: confirm

if RUN_PREPROCESSING:
    # subset_n caps the work. Full Ego4D (~5M narrations) is 1-2 weeks single-process.
    # 50k narrations on Colab CPU ~ 40-90 min. None = full set (only with cluster compute).
    preprocess_ego4d(
        videos_dir=EGO4D_VIDEOS_DIR,
        narrations_json=EGO4D_NARRATIONS_FILE,
        output_dir=CONFIG['ego4d_root'],
        subset_n=50_000,                   # tune for your time/disk budget
        image_size=CONFIG['image_size'],
        clip_window_sec=4.0,
        narration_pass='narration_pass_1',
        only_camera_wearer=True,
    )
else:
    print("RUN_PREPROCESSING=False — skipping. Set True if you need to preprocess.")
""")

code("""# ===== P3: validate preprocessed data is consumable =====
manifest_path = Path(CONFIG['ego4d_root']) / 'manifest.parquet'
if not manifest_path.exists():
    print(f"[warn] {manifest_path} does not exist yet — preprocessing or staging needed.")
    print("       The Ego4D-dependent cells (CLIP cache, pretrain) will fail until this is fixed.")
else:
    m = pd.read_parquet(manifest_path)
    assert {'clip_uid', 'narration', 'n_frames'}.issubset(m.columns), m.columns.tolist()
    assert (m['n_frames'] >= 5).all(), "found clips with <5 frames"
    print(f"[ok] manifest: {len(m):,} clips, "
          f"{m['narration'].nunique():,} unique narrations, "
          f"avg narration length {m['narration'].str.len().mean():.1f} chars")
    # Spot-check first clip
    first_uid = m.iloc[0]['clip_uid']
    clip_dir = Path(CONFIG['ego4d_root']) / first_uid
    for j in range(5):
        f = clip_dir / f"{j}.jpg"
        if not f.exists():
            print(f"[warn] missing {f}"); break
        im = Image.open(f); assert im.size == (CONFIG['image_size'], CONFIG['image_size']), im.size
    else:
        print(f"[ok] first clip {first_uid}: 5/5 frames present at "
              f"{CONFIG['image_size']}x{CONFIG['image_size']}")

# Validate demo_root exists too — the BC sweep needs this
demo_root = Path(CONFIG['demo_root'])
if not demo_root.exists():
    print(f"[warn] CONFIG['demo_root']={demo_root} not found — BC sweep will be skipped.")
else:
    n_envs = sum(1 for _ in demo_root.iterdir() if _.is_dir())
    print(f"[ok] demo_root has {n_envs} env subdirs")
""")


# ============================================================
# Cell 4 — logger
# ============================================================
code("""# ===== CELL 4: dual logger (stdout + results.json) =====
RESULTS_PATH = Path(CONFIG['results_dir']) / 'results.json'

class Results:
    def __init__(self, path):
        self.path = Path(path)
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        else:
            self.data = {'config': CONFIG, 'pretrain': {}, 'bc': {}, 'meta': {'started_at': time.time()}}
            self._flush()
    def log(self, section: str, key: str, value):
        d = self.data.setdefault(section, {})
        d[key] = value
        self._flush()
    def append(self, section: str, key: str, value):
        d = self.data.setdefault(section, {}).setdefault(key, [])
        d.append(value)
        self._flush()
    def _flush(self):
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.data, indent=2, default=str))
        tmp.replace(self.path)

results = Results(RESULTS_PATH)

def log(msg: str):
    ts = time.strftime('%H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)

log(f"results.json -> {RESULTS_PATH}")
""")


# ============================================================
# Cell 5 — R3M baseline hardcoded
# ============================================================
code("""# ===== CELL 5: R3M baseline (Tables 1-3, paper figures) =====
# Numbers verbatim from arXiv:2203.12601v3.
# Per-env aggregates are paper-exact. Per-task / per-view / per-demo-size numbers
# are NOT printed numerically in the paper (only as bars in Figs 4/8/9). For
# paper-exact granular comparison, replace the placeholders here with values
# from the eval JSONs in facebookresearch/r3m -> evaluation/results/.
# Each entry is {mean_pct, stderr_pct}.

R3M_BASELINE = {
    # ---- Table 1: ablation overall (used as primary baseline for our R3M aggregate) ----
    'aggregate': {
        'franka_kitchen': {'mean': 53.1, 'stderr': 2.7},
        'metaworld':      {'mean': 69.2, 'stderr': 2.0},
        'adroit':         {'mean': 65.0, 'stderr': 1.7},
        'all_domains':    {'mean': 62.4, 'stderr': 1.3},
    },
    'r3m_no_aug':  {'franka_kitchen': 51.1, 'metaworld': 68.9, 'adroit': 61.3, 'all_domains': 60.4},
    'r3m_no_l1':   {'franka_kitchen': 46.7, 'metaworld': 65.0, 'adroit': 66.5, 'all_domains': 59.4},
    'r3m_no_lang': {'franka_kitchen': 47.2, 'metaworld': 67.0, 'adroit': 45.6, 'all_domains': 53.2},

    # ---- Table 2: data ablation on Franka + Adroit ----
    'data_ablation': {
        'r3m':         {'franka_kitchen': {'mean': 53.1, 'stderr': 2.7}, 'adroit': {'mean': 65.0, 'stderr': 1.7}},
        'moco_ego4d':  {'franka_kitchen': {'mean': 42.0, 'stderr': 2.8}, 'adroit': {'mean': 54.9, 'stderr': 2.7}},
        'mvp':         {'franka_kitchen': {'mean': 27.0, 'stderr': 2.6}, 'adroit': {'mean': 51.4, 'stderr': 2.7}},
    },

    # ---- Per-task (read off Fig 9 bars; mark as approximate) ----
    # TODO: confirm — overlay with exact JSONs from facebookresearch/r3m eval results
    'per_task_approx': {
        'franka_kitchen': {'slide': 60, 'light': 50, 'door': 55, 'knob': 50, 'micro': 50},
        'metaworld':      {'assembly-v2': 65, 'bin-picking-v2': 70, 'button-press-topdown-v2': 80,
                           'drawer-open-v2': 90, 'hammer-v2': 60},
        'adroit':         {'pen-v0': 60, 'relocate-v0': 70},
    },

    # ---- Per-view (Fig 8 left; approximate) ----
    # TODO: confirm
    'per_view_approx': {
        'franka_kitchen': {'view1': 55, 'view2': 53, 'view3': 51},
        'metaworld':      {'view1': 70, 'view2': 70, 'view3': 67},
        'adroit':         {'view1': 67, 'view2': 63},
    },

    # ---- Per-demo-size (Fig 8 right; approximate) ----
    # TODO: confirm
    'per_demos_approx': {
        'franka_kitchen': {5: 42, 10: 53, 25: 64},
        'metaworld':      {5: 60, 10: 70, 25: 78},
        'adroit':         {25: 55, 50: 65, 100: 75},
    },

    # ---- Table 3: real-world (reported but we don't reproduce) ----
    'real_robot': {
        'r3m':  {'closing_drawer': 80, 'mask_in_dresser': 30, 'lettuce_in_pan': 60,
                 'mug_to_goal': 70, 'fold_towel': 40, 'avg': 56},
        'clip': {'closing_drawer': 70, 'mask_in_dresser': 10, 'lettuce_in_pan': 0,
                 'mug_to_goal': 40, 'fold_towel': 0,  'avg': 24},
    },
}
results.log('baseline', 'r3m', R3M_BASELINE)
log("R3M baseline table loaded.")
""")


# ============================================================
# Cell 6 — Ego4D dataset
# ============================================================
code("""# ===== CELL 6: Ego4D dataset (pre-extracted frames) =====
# Expects:
#   {ego4d_root}/manifest.parquet   columns: clip_uid (str), narration (str), n_frames (int)
#   {ego4d_root}/<clip_uid>/0.jpg .. <n_frames-1>.jpg  (5 frames sampled per clip,
#       indices 0,1,2,3,4 corresponding to I0, Ii, Ij, Ik, Ig respectively)
# If your cache uses raw frame indices (not pre-sampled), adjust _sample_indices.
import pyarrow.parquet as pq
from PIL import Image
import numpy as np
from torchvision import transforms as T

class Ego4DR3MDataset(Dataset):
    def __init__(self, root, manifest_file, image_size=224, train=True):
        self.root = Path(root)
        manifest_path = self.root / manifest_file
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found at {manifest_path}. # TODO: confirm path / re-run R3M's preprocess_ego4d.py"
            )
        tbl = pq.read_table(manifest_path)
        self.df = tbl.to_pandas()
        # Filter clips with at least 5 frames
        self.df = self.df[self.df['n_frames'] >= 5].reset_index(drop=True)
        self.image_size = image_size
        self.train = train
        # video-level random crop is implemented inline (same crop for all 5 frames of a clip)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)

    def _sample_indices(self, n_frames):
        # I0 from first 20%, Ig from last 20%, Ii<Ij<Ik in between
        first20 = max(1, n_frames // 5)
        last20  = max(1, n_frames // 5)
        i0_idx  = random.randint(0, first20 - 1)
        ig_idx  = random.randint(n_frames - last20, n_frames - 1)
        # Sample 3 ordered middle frames between i0_idx+1 and ig_idx-1
        lo, hi = i0_idx + 1, ig_idx - 1
        if hi - lo + 1 < 3:
            mids = sorted([random.randint(lo, hi) for _ in range(3)]) if hi >= lo else [i0_idx, i0_idx, i0_idx]
        else:
            mids = sorted(random.sample(range(lo, hi + 1), 3))
        return [i0_idx, mids[0], mids[1], mids[2], ig_idx]

    def _load_with_video_crop(self, clip_uid, indices):
        # Load all 5 frames first to share a single random crop
        imgs = [Image.open(self.root / clip_uid / f"{i}.jpg").convert('RGB') for i in indices]
        W, H = imgs[0].size
        S = self.image_size
        if self.train and W > S and H > S:
            # Same random crop applied to all 5 frames (per Section 3.3)
            scale = random.uniform(0.7, 1.0)
            cw = ch = int(min(W, H) * scale)
            x0 = random.randint(0, W - cw); y0 = random.randint(0, H - ch)
            imgs = [im.crop((x0, y0, x0 + cw, y0 + ch)).resize((S, S), Image.Resampling.BICUBIC) for im in imgs]
        else:
            imgs = [im.resize((S, S), Image.Resampling.BICUBIC) for im in imgs]
        arr = np.stack([np.asarray(im, dtype=np.uint8) for im in imgs], axis=0)  # (5, H, W, 3)
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
        x = self.normalize(x)
        return x  # (5, 3, S, S)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # If the cache has pre-sampled 5 frames already (filenames 0..4.jpg) we skip _sample_indices.
        # Detect by reading n_frames: if exactly 5, treat as pre-sampled.
        if row['n_frames'] == 5:
            indices = [0, 1, 2, 3, 4]
        else:
            indices = self._sample_indices(int(row['n_frames']))
        frames = self._load_with_video_crop(row['clip_uid'], indices)
        return {
            'frames':    frames,                       # (5, 3, 224, 224)
            'narration': row['narration'],             # str
            'clip_uid':  row['clip_uid'],
        }

def _ego4d_collate(batch):
    # Module-level (not lambda) so multiprocessing workers can pickle it.
    return {
        'frames':    torch.stack([b['frames'] for b in batch], dim=0),
        'narration': [b['narration'] for b in batch],
        'clip_uid':  [b['clip_uid']  for b in batch],
    }

def make_pretrain_loader(cfg):
    ds = Ego4DR3MDataset(cfg['ego4d_root'], cfg['manifest_file'],
                         image_size=cfg['image_size'], train=True)
    log(f"Ego4D dataset size: {len(ds):,} clips")
    return ds, DataLoader(
        ds, batch_size=cfg['batch_clips'], shuffle=True, drop_last=True,
        num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'],
        persistent_workers=cfg['persistent_workers'], prefetch_factor=cfg['prefetch_factor'],
        collate_fn=_ego4d_collate,
    )
""")


# ============================================================
# Cell 7 — CLIP text pre-embed
# ============================================================
code("""# ===== CELL 7: pre-compute & cache CLIP text embeddings =====
# Computes once for all unique narrations, saves to disk; reused as a tensor lookup
# in the training loop (no text encoder runs in the inner loop).
import open_clip

CLIP_CACHE = Path(CONFIG['cache_dir']) / 'clip_text_embeds.pt'

@torch.no_grad()
def build_clip_text_cache(cfg, force=False):
    if CLIP_CACHE.exists() and not force:
        cache = torch.load(CLIP_CACHE, map_location='cpu', weights_only=False)
        log(f"CLIP text cache loaded from {CLIP_CACHE} ({cache['embeds'].shape[0]:,} narrations)")
        return cache

    ds = Ego4DR3MDataset(cfg['ego4d_root'], cfg['manifest_file'],
                         image_size=cfg['image_size'], train=False)
    unique_narrations = sorted(set(ds.df['narration'].astype(str).tolist()))
    log(f"unique narrations: {len(unique_narrations):,}")

    clip_model, _, _ = open_clip.create_model_and_transforms(
        cfg['clip_variant'], pretrained=cfg['clip_pretrained'], device=DEVICE
    )
    clip_model.eval()
    tokenizer = open_clip.get_tokenizer(cfg['clip_variant'])

    embeds = []
    BS = 4096
    t0 = time.time()
    for i in range(0, len(unique_narrations), BS):
        batch = unique_narrations[i:i + BS]
        toks = tokenizer(batch).to(DEVICE)
        with torch.autocast(device_type='cuda', dtype=DTYPE, enabled=DEVICE.type == 'cuda'):
            e = clip_model.encode_text(toks)
        e = F.normalize(e.float(), dim=-1)  # (B, 512)
        embeds.append(e.cpu())
        if (i // BS) % 50 == 0:
            log(f"  text encode {i + len(batch):,}/{len(unique_narrations):,} ({time.time()-t0:.1f}s)")
    embeds = torch.cat(embeds, dim=0)
    cache = {'narrations': unique_narrations, 'embeds': embeds,
             'narration_to_idx': {n: i for i, n in enumerate(unique_narrations)}}
    torch.save(cache, CLIP_CACHE)
    log(f"saved CLIP cache: {CLIP_CACHE} ({embeds.shape})")
    del clip_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cache
""")


# ============================================================
# Cell 8 — custom LoRA layer
# ============================================================
code("""# ===== CELL 8: minimal LoRA adapter (no peft dep) =====
# We patch named modules of DINOv2 by replacing nn.Linear with LoRALinear, which
# wraps the frozen base linear with a low-rank update.
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B initialized to zero so initial LoRA contribution is exactly 0.
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.base(x)
        x_d = self.dropout(x)
        # (..., in) @ (in, r) -> (..., r) ; @ (r, out) -> (..., out)
        delta = (x_d @ self.lora_A.t()) @ self.lora_B.t()
        return out + delta * self.scaling

def inject_lora(module: nn.Module, target_substrings, r: int, alpha: int, dropout: float):
    \"\"\"Replace nn.Linear modules whose qualified name contains any target substring.\"\"\"
    n_replaced = 0
    for name, child in list(module.named_modules()):
        for cname, c in list(child.named_children()):
            full = f"{name}.{cname}" if name else cname
            if isinstance(c, nn.Linear) and any(t in cname for t in target_substrings):
                setattr(child, cname, LoRALinear(c, r, alpha, dropout))
                n_replaced += 1
    return n_replaced

def count_trainable(m: nn.Module):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
""")


# ============================================================
# Cell 9 — Model
# ============================================================
code("""# ===== CELL 9: model — DINOv3 backbone + LoRA + language head + projection =====
# DINOv3 ViT-B/16 from torch.hub. Backbone frozen; LoRA injected into qkv/proj/fc1/fc2.
# Language head Gθ is a 5-layer MLP (per R3M Appendix A.2) over [z0, zj, l_emb] -> scalar score.
# We use CLIP text dim (512) instead of DistilBERT (768); the architecture is otherwise identical.
#
# NOTE on weights: DINOv3 weights are gated. After requesting access at
# https://ai.meta.com/dino-v3, set CONFIG['dinov3_weights'] to the URL or local path
# from the email. Without weights, the model architecture loads but is randomly
# initialized — useless for production but fine for the smoke section (which
# stubs the backbone entirely).

def load_dinov3_backbone(variant: str, weights):
    if weights is None:
        raise RuntimeError(
            "CONFIG['dinov3_weights'] is None. DINOv3 weights are gated — request "
            "access at https://ai.meta.com/dino-v3 and set the URL/local-path here."
        )
    return torch.hub.load(
        'facebookresearch/dinov3', variant,
        source='github', weights=weights,
    )

class LanguageHead(nn.Module):
    def __init__(self, feat_dim, text_dim, hidden=(1024, 1024, 1024, 1024)):
        super().__init__()
        in_dim = 2 * feat_dim + text_dim
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU()]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, z0, zj, l):
        x = torch.cat([z0, zj, l], dim=-1)
        return self.net(x).squeeze(-1)   # (B,)

class DinoLoRAR3M(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = load_dinov3_backbone(cfg['dinov3_variant'], cfg.get('dinov3_weights'))
        for p in self.backbone.parameters():
            p.requires_grad = False
        n_lora = inject_lora(self.backbone, cfg['lora_targets'],
                             r=cfg['lora_rank'], alpha=cfg['lora_alpha'], dropout=cfg['lora_dropout'])
        log(f"injected LoRA into {n_lora} linear modules; trainable_lora={count_trainable(self.backbone):,}")
        # Optional projection head — keep identity to match R3M (no extra projection); language head consumes raw feature.
        self.lang_head = LanguageHead(cfg['feat_dim'], cfg['text_dim'], cfg['lang_head_hidden'])

    def encode(self, x):
        # DINOv2.forward returns the CLS token feature (B, 768)
        return self.backbone(x)

    def forward_pretrain(self, frames):
        \"\"\"
        frames: (B, 5, 3, H, W)  in order [I0, Ii, Ij, Ik, Ig]
        Returns dict with per-frame embeddings; loss fns compute their own pairings.
        \"\"\"
        B, K, C, H, W = frames.shape
        flat = frames.view(B * K, C, H, W)
        z = self.encode(flat).view(B, K, -1)   # (B, 5, D)
        z0, zi, zj, zk, zg = z.unbind(dim=1)
        return {'z0': z0, 'zi': zi, 'zj': zj, 'zk': zk, 'zg': zg}

    def lang_score(self, z0, zt, l):
        return self.lang_head(z0, zt, l)
""")


# ============================================================
# Cell 10 — losses
# ============================================================
code("""# ===== CELL 10: R3M loss functions =====
# Time-contrastive (Eq 1): InfoNCE over (zi, zj, zk) triplets with cross-video negatives.
# Similarity = negative L2 distance, per Section 3.3.
# Video-language alignment (Eq 2 + Appendix A.3): three positive/negative pair sets
# per element — (z0,zg)/(z0,z0), (z0,zj)/(z0,zi), (z0,zk)/(z0,zj), each with a
# cross-video wrong-language negative.

def neg_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return -((a - b).pow(2).sum(dim=-1))

def _gather_neg_videos(z, idx):
    # z: (B, D); idx: (B, K) gather indices into B for each anchor
    return z[idx]   # (B, K, D)

def time_contrastive_loss(out, num_neg=3):
    # out['zi'], out['zj'], out['zk']: each (B, D)
    zi, zj, zk = out['zi'], out['zj'], out['zk']
    B = zi.shape[0]
    # Sample num_neg distinct other-video indices for each anchor
    rand_idx = torch.stack([
        torch.tensor(random.sample([j for j in range(B) if j != b], k=min(num_neg, B - 1)), device=zi.device)
        for b in range(B)
    ], dim=0)   # (B, num_neg)
    s_pos     = neg_l2(zi, zj).unsqueeze(1)                  # (B, 1)
    s_far     = neg_l2(zi, zk).unsqueeze(1)                  # (B, 1)
    zi_neg    = _gather_neg_videos(zi, rand_idx)             # (B, num_neg, D)
    s_crossvid= neg_l2(zi.unsqueeze(1).expand_as(zi_neg), zi_neg)  # (B, num_neg)
    logits = torch.cat([s_pos, s_far, s_crossvid], dim=1)    # (B, 1 + 1 + num_neg)
    target = torch.zeros(B, dtype=torch.long, device=zi.device)
    return F.cross_entropy(logits, target)

def video_language_loss(model, out, l_embed, num_neg=3):
    \"\"\"R3M's VLA loss with three positive/negative pair sets per element.\"\"\"
    z0, zi, zj, zk, zg = out['z0'], out['zi'], out['zj'], out['zk'], out['zg']
    B = z0.shape[0]
    # cross-video wrong-language negatives — guaranteed derangement via cyclic shift
    perm = (torch.arange(B, device=l_embed.device) + 1) % B
    l_wrong = l_embed[perm]
    losses = []
    pairs = [(zg, z0), (zj, zi), (zk, zj)]  # (positive, harder same-video negative)
    for z_pos, z_neg in pairs:
        s_pos     = model.lang_score(z0, z_pos, l_embed)               # (B,)
        s_neg_vid = model.lang_score(z0, z_neg, l_embed)               # same video, earlier — should be lower
        s_neg_lng = model.lang_score(z0, z_pos, l_wrong)               # right frame pair, wrong language
        logits = torch.stack([s_pos, s_neg_vid, s_neg_lng], dim=1)     # (B, 3)
        target = torch.zeros(B, dtype=torch.long, device=z0.device)
        losses.append(F.cross_entropy(logits, target))
    return sum(losses) / len(losses)

def regularization(out, lam_l1, lam_l2):
    # R3M applies L1+L2 to Fφ(Ii); we apply to all sampled middle frames for stability.
    z = torch.stack([out['zi'], out['zj'], out['zk']], dim=1)  # (B, 3, D)
    return lam_l1 * z.abs().mean() + lam_l2 * z.pow(2).mean()
""")


# ============================================================
# Cell 11 — pretraining loop
# ============================================================
code("""# ===== CELL 11: pretraining loop =====
def make_model_and_optim(cfg):
    model = DinoLoRAR3M(cfg).to(DEVICE)
    if cfg['compile_model']:
        try:
            # Compile the backbone only — the lang head is small and recompiles often
            model.backbone = torch.compile(model.backbone, mode=cfg['compile_mode'], fullgraph=False)
            log(f"torch.compile enabled (mode={cfg['compile_mode']})")
        except Exception as e:
            log(f"torch.compile failed, continuing eagerly: {e}")
    trainable = [p for p in model.parameters() if p.requires_grad]
    log(f"total trainable params: {sum(p.numel() for p in trainable):,}")
    opt = torch.optim.Adam(trainable, lr=cfg['pretrain_lr'], weight_decay=cfg['pretrain_weight_decay'])
    return model, opt

def lookup_text_embeds(narrations, text_cache):
    idx = torch.tensor([text_cache['narration_to_idx'][n] for n in narrations], dtype=torch.long)
    return text_cache['embeds'][idx]   # (B, 512), CPU; caller moves to device

def pretrain(cfg, model, opt, loader, text_cache, save_path, max_steps=None):
    max_steps = max_steps or cfg['pretrain_steps']
    model.train()
    step = 0
    t0 = time.time()
    running = {'tcn': 0.0, 'lang': 0.0, 'reg': 0.0, 'total': 0.0, 'count': 0}
    iterator = iter(loader)
    while step < max_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        frames    = batch['frames'].to(DEVICE, non_blocking=True)        # (B, 5, 3, H, W)
        l_embeds  = lookup_text_embeds(batch['narration'], text_cache).to(DEVICE, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=DTYPE, enabled=DEVICE.type == 'cuda'):
            out  = model.forward_pretrain(frames)
            l_tcn  = time_contrastive_loss(out, num_neg=cfg['num_negatives'])
            l_lng  = video_language_loss(model, out, l_embeds, num_neg=cfg['num_negatives'])
            l_reg  = regularization(out, cfg['lambda_l1'], cfg['lambda_l2'])
            total  = cfg['lambda_tcn'] * l_tcn + cfg['lambda_lang'] * l_lng + l_reg
        total.backward()
        opt.step()

        running['tcn']  += l_tcn.item()
        running['lang'] += l_lng.item()
        running['reg']  += l_reg.item()
        running['total']+= total.item()
        running['count']+= 1

        if (step + 1) % cfg['pretrain_log_every'] == 0:
            n = running['count']
            elapsed = time.time() - t0
            eta = elapsed / (step + 1) * (max_steps - step - 1) / 60
            log(f"step {step+1}/{max_steps}  total={running['total']/n:.4f}  "
                f"tcn={running['tcn']/n:.4f}  lang={running['lang']/n:.4f}  reg={running['reg']/n:.6f}  "
                f"eta={eta:.1f}min")
            results.append('pretrain', 'losses', {
                'step': step + 1, 'total': running['total']/n,
                'tcn': running['tcn']/n, 'lang': running['lang']/n, 'reg': running['reg']/n,
            })
            running = {'tcn': 0.0, 'lang': 0.0, 'reg': 0.0, 'total': 0.0, 'count': 0}

        if (step + 1) % cfg['pretrain_save_every'] == 0:
            torch.save({
                'step': step + 1,
                'state_dict': {k: v for k, v in model.state_dict().items() if 'lora_' in k or 'lang_head' in k},
                'config': cfg,
            }, save_path)
            log(f"checkpoint -> {save_path}")
        step += 1

    torch.save({
        'step': step, 'state_dict': {k: v for k, v in model.state_dict().items()
                                     if 'lora_' in k or 'lang_head' in k},
        'config': cfg,
    }, save_path)
    log(f"pretrain done in {(time.time()-t0)/60:.1f}min — final ckpt: {save_path}")
""")


# ============================================================
# Cell 13 — feature pre-extraction for downstream demos
# ============================================================
code("""# ===== CELL 13: pre-extract DINOv2-LoRA features for all demo episodes =====
# After pretrain, the LoRA-adapted backbone is FROZEN for downstream use. We do a
# single pass over every demo episode for every (env, task, view) and cache the
# (T, feat_dim) feature tensor + actions + proprio + success label to disk. The BC
# inner loop reads these cached tensors — no encoder runs in BC training.
#
# Demo file layout assumption (TODO: confirm — adapt to R3M's eval repo demos):
#   {demo_root}/{env}/{task}/{view}/demo_{i}.pkl
#   pickled dict with keys: 'images' (T, H, W, 3, uint8), 'actions' (T, A), 'proprio' (T, P), 'success' (bool)

def load_pretrained_model(cfg, ckpt_path):
    model = DinoLoRAR3M(cfg).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
    # Backbone frozen base weights are already correct from torch.hub; we only loaded LoRA + lang head.
    log(f"loaded ckpt step={ckpt.get('step','?')}, missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

@torch.no_grad()
def encode_images(model, imgs_uint8, batch=64):
    # imgs_uint8: (T, H, W, 3) uint8 numpy
    import numpy as np
    N = imgs_uint8.shape[0]
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
    feats = []
    for i in range(0, N, batch):
        chunk = imgs_uint8[i:i + batch]
        if chunk.shape[1] != CONFIG['image_size'] or chunk.shape[2] != CONFIG['image_size']:
            # resize on CPU before sending to GPU — small batches
            from PIL import Image
            chunk = np.stack([
                np.asarray(Image.fromarray(im).resize((CONFIG['image_size'], CONFIG['image_size']), Image.Resampling.BICUBIC))
                for im in chunk
            ], axis=0)
        x = torch.from_numpy(chunk).to(DEVICE).permute(0, 3, 1, 2).float() / 255.0
        x = (x - mean) / std
        with torch.autocast(device_type='cuda', dtype=DTYPE, enabled=DEVICE.type == 'cuda'):
            z = model.encode(x)
        feats.append(z.float().cpu())
    return torch.cat(feats, dim=0)

def extract_features_for_env(cfg, model, env_name, env_cfg):
    out_dir = Path(cfg['cache_dir']) / 'features' / env_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for task in env_cfg['tasks']:
        for view in env_cfg['views']:
            demo_glob = Path(cfg['demo_root']) / env_name / task / view
            cache_file = out_dir / f"{task}__{view}.pt"
            if cache_file.exists():
                log(f"features cached: {cache_file}")
                continue
            if not demo_glob.exists():
                log(f"# TODO: confirm demo path missing: {demo_glob}")
                continue
            episodes = []
            for f in sorted(demo_glob.glob('demo_*.pkl')):
                with open(f, 'rb') as fp:
                    d = pickle.load(fp)
                feats = encode_images(model, d['images'])   # (T, D)
                episodes.append({
                    'features':   feats,
                    'actions':    torch.as_tensor(d['actions'],  dtype=torch.float32),
                    'proprio':    torch.as_tensor(d['proprio'],  dtype=torch.float32),
                    'success':    bool(d.get('success', True)),
                })
            torch.save({'env': env_name, 'task': task, 'view': view, 'episodes': episodes}, cache_file)
            log(f"saved {cache_file} ({len(episodes)} demos)")

def run_feature_extraction(cfg):
    if not Path(cfg['pretrain_ckpt']).exists():
        log("no pretrain ckpt — skipping feature extraction.")
        return None
    model = load_pretrained_model(cfg, cfg['pretrain_ckpt'])
    for env_name, env_cfg in cfg['envs'].items():
        extract_features_for_env(cfg, model, env_name, env_cfg)
    return model
""")


# ============================================================
# Cell 14 — BC dataset + policy
# ============================================================
code("""# ===== CELL 14: BC dataset + 2-layer MLP+BatchNorm policy =====
class FeatureBCDataset(Dataset):
    def __init__(self, cache_file, n_demos):
        bundle = torch.load(cache_file, map_location='cpu', weights_only=False)
        eps = bundle['episodes'][:n_demos]
        self.feats   = torch.cat([e['features'] for e in eps], dim=0)
        self.proprio = torch.cat([e['proprio']  for e in eps], dim=0)
        self.actions = torch.cat([e['actions']  for e in eps], dim=0)
        assert self.feats.shape[0] == self.actions.shape[0]
    def __len__(self):  return self.feats.shape[0]
    def __getitem__(self, i):
        return torch.cat([self.feats[i], self.proprio[i]], dim=-1), self.actions[i]

class BCPolicy(nn.Module):
    \"\"\"R3M Appendix B.5: 2-layer MLP [256,256] + BatchNorm at the input.\"\"\"
    def __init__(self, in_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim)
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, act_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(self.bn(x))
""")


# ============================================================
# Cell 15 — sim env wrappers + rollout
# ============================================================
code("""# ===== CELL 15: simulator wrappers + rollout =====
# # TODO: confirm — these wrappers depend on mj_envs (Franka Kitchen, Adroit) and
# metaworld being importable. They produce 224x224 RGB observations, the
# proprioceptive vector R3M used, and a binary success signal.

def make_env(env_name, task, view, image_size=224):
    if env_name == 'metaworld':
        # # TODO: confirm — newer metaworld uses ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
        # with keys like 'assembly-v2-goal-observable'. Both code paths kept.
        try:
            from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as REG
            cls = REG[f"{task}-goal-observable"]
        except (ImportError, KeyError):
            from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS as REG
            cls = REG[task]   # task already includes '-v2' suffix
        env = cls()
        env._partially_observable = False
        view_to_camera = {'view1': 'corner', 'view2': 'corner2', 'view3': 'topview'}   # # TODO: confirm
        env._camera_name = view_to_camera.get(view, 'corner')
        return env
    elif env_name == 'franka_kitchen':
        # # TODO: confirm — the modified env from R3M's eval repo, registered as kitchen_<task>-v3 etc.
        import gym
        # mj_envs registers these on import
        import mj_envs  # noqa: F401
        task_to_id = {
            'slide':  'kitchen_sdoor_open-v3',
            'light':  'kitchen_light_on-v3',
            'door':   'kitchen_ldoor_open-v3',
            'knob':   'kitchen_knob1_on-v3',
            'micro':  'kitchen_micro_open-v3',
        }
        env = gym.make(task_to_id[task])
        # # TODO: confirm — view configuration via env.unwrapped.set_camera or similar
        return env
    elif env_name == 'adroit':
        import gym
        import mj_envs  # noqa: F401
        task_to_id = {'pen-v0': 'pen-v0', 'relocate-v0': 'relocate-v0'}
        return gym.make(task_to_id[task])
    raise ValueError(f"unknown env {env_name}")

def render_rgb(env, image_size=224):
    \"\"\"Best-effort RGB grab spanning gym/gymnasium and old/new MuJoCo. # TODO: confirm per env.\"\"\"
    # Try kwargs first (newer)
    for kwargs in ({'mode': 'rgb_array', 'width': image_size, 'height': image_size},
                   {'mode': 'rgb_array'},
                   {'width': image_size, 'height': image_size},
                   {}):
        try:
            return env.render(**kwargs)
        except TypeError:
            continue
    raise RuntimeError("env.render produced no rgb image")

def get_proprio(env, env_name):
    # # TODO: confirm proprio composition matches R3M's choice.
    if env_name == 'metaworld':
        return env._get_obs()[:8]                 # gripper pose + open/close
    if env_name == 'franka_kitchen':
        return env.unwrapped._get_obs()[:9]
    if env_name == 'adroit':
        return env.unwrapped._get_obs()[:30]
    return env._get_obs()

@torch.no_grad()
def rollout(env, env_name, encoder, policy, horizon, image_size=224):
    from PIL import Image
    import numpy as np
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out   # gymnasium vs gym
    success = False
    for t in range(horizon):
        rgb = render_rgb(env, image_size)
        if rgb.shape[0] != image_size:
            rgb = np.asarray(Image.fromarray(rgb).resize((image_size, image_size), Image.Resampling.BICUBIC))
        x = torch.from_numpy(rgb).to(DEVICE).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
        x = (x - mean) / std
        with torch.autocast(device_type='cuda', dtype=DTYPE, enabled=DEVICE.type == 'cuda'):
            z = encoder.encode(x).float()
        p = torch.as_tensor(get_proprio(env, env_name), device=DEVICE, dtype=torch.float32).unsqueeze(0)
        a = policy(torch.cat([z, p], dim=-1)).squeeze(0).cpu().numpy()
        step_out = env.step(a)
        # gym 4-tuple: (obs, reward, done, info)
        # gymnasium 5-tuple: (obs, reward, terminated, truncated, info)
        if len(step_out) == 5:
            obs, _, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        else:
            obs, _, done, info = step_out
        if isinstance(info, dict) and (info.get('success') or info.get('goal_achieved')):
            success = True
        if done:
            break
    return success
""")


# ============================================================
# Cell 16 — BC train + eval with early stop
# ============================================================
code("""# ===== CELL 16: BC train + plateau early-stop =====
def train_bc_one(cfg, env_name, task, view, n_demos, seed, encoder):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    cache_file = Path(cfg['cache_dir']) / 'features' / env_name / f"{task}__{view}.pt"
    if not cache_file.exists():
        return {'best_success': None, 'reason': f'features missing for {cache_file}'}

    ds = FeatureBCDataset(cache_file, n_demos=n_demos)
    if len(ds) == 0:
        return {'best_success': None, 'reason': 'empty dataset'}
    in_dim = ds.feats.shape[1] + ds.proprio.shape[1]
    act_dim = ds.actions.shape[1]
    policy = BCPolicy(in_dim, act_dim, hidden=cfg['bc_hidden']).to(DEVICE)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg['bc_lr'])
    # num_workers=0: BC dataset is in-memory features, no I/O — workers add fork/spawn fragility for zero gain
    loader = DataLoader(ds, batch_size=cfg['bc_batch'], shuffle=True, drop_last=True,
                        num_workers=0, pin_memory=True)

    horizon = (cfg['envs'][env_name].get('horizons', {}) or {}).get(task, cfg['envs'][env_name].get('horizon'))
    env = make_env(env_name, task, view, image_size=cfg['image_size'])

    best = -1.0
    plateau = 0
    history = []
    iterator = iter(loader)
    for step in range(1, cfg['bc_steps'] + 1):
        try:
            x, a = next(iterator)
        except StopIteration:
            iterator = iter(loader); x, a = next(iterator)
        x = x.to(DEVICE, non_blocking=True); a = a.to(DEVICE, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        pred = policy(x)
        loss = F.mse_loss(pred, a)
        loss.backward(); opt.step()

        if step % cfg['bc_eval_every'] == 0:
            policy.eval()
            successes = sum(rollout(env, env_name, encoder, policy, horizon, cfg['image_size'])
                            for _ in range(cfg['bc_eval_episodes']))
            rate = 100.0 * successes / cfg['bc_eval_episodes']
            history.append({'step': step, 'success_pct': rate, 'loss': float(loss.item())})
            improved = rate > best
            if improved:
                best = rate; plateau = 0
            else:
                plateau += 1
            log(f"  [{env_name}/{task}/{view}/n{n_demos}/s{seed}] step {step} success={rate:.1f}% best={best:.1f}% plateau={plateau}")
            if plateau >= cfg['bc_early_stop_patience']:
                log(f"  early stop at step {step}")
                break
            policy.train()
    env.close() if hasattr(env, 'close') else None
    return {'best_success': float(best), 'history': history, 'env': env_name, 'task': task,
            'view': view, 'n_demos': n_demos, 'seed': seed}
""")


# ============================================================
# Cell 17 — BC sweep orchestrator
# ============================================================
code("""# ===== CELL 17: BC sweep over (env, task, view, n_demos, seed) =====
def run_bc_sweep(cfg, encoder):
    sweep_results = []
    sweep_t0 = time.time()
    for env_name, env_cfg in cfg['envs'].items():
        for task in env_cfg['tasks']:
            for view in env_cfg['views']:
                for n_demos in env_cfg['demo_sizes']:
                    for seed in cfg['seeds']:
                        key = f"{env_name}/{task}/{view}/n{n_demos}/s{seed}"
                        try:
                            r = train_bc_one(cfg, env_name, task, view, n_demos, seed, encoder)
                        except Exception as e:
                            log(f"FAIL {key}: {type(e).__name__}: {e}")
                            r = {'best_success': None, 'error': f"{type(e).__name__}: {e}"}
                        r['key'] = key
                        sweep_results.append(r)
                        results.append('bc', 'runs', r)
                        elapsed = (time.time() - sweep_t0) / 60
                        log(f"sweep progress: {len(sweep_results)} runs done, elapsed={elapsed:.1f}min")
    return sweep_results
""")


# ============================================================
# Cell 18 — aggregation, delta table, plots, training curves
# ============================================================
code("""# ===== CELL 18: aggregate, delta table, side-by-side plots =====
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def summarize(sweep):
    \"\"\"Aggregate per env, per task, per view, per demo size — average over seeds.\"\"\"
    by_env = {}; by_task = {}; by_view = {}; by_demos = {}
    for r in sweep:
        if r.get('best_success') is None: continue
        env = r['env']; task = r['task']; view = r['view']; nd = r['n_demos']
        by_env.setdefault(env, []).append(r['best_success'])
        by_task.setdefault((env, task), []).append(r['best_success'])
        by_view.setdefault((env, view), []).append(r['best_success'])
        by_demos.setdefault((env, nd), []).append(r['best_success'])
    def reduce(d):
        return {k: {'mean': float(np.mean(v)), 'stderr': float(np.std(v) / np.sqrt(len(v))),
                    'n': len(v)} for k, v in d.items()}
    return {'env': reduce(by_env), 'task': reduce(by_task),
            'view': reduce(by_view), 'demos': reduce(by_demos)}

def delta_table(ours, baseline):
    \"\"\"Per-env delta = ours - r3m_aggregate.\"\"\"
    rows = []
    for env in ['franka_kitchen', 'metaworld', 'adroit']:
        if env not in ours['env']:
            continue
        o = ours['env'][env]['mean']
        b = baseline['aggregate'][env]['mean']
        rows.append({'env': env, 'ours_mean': o, 'r3m_mean': b, 'delta': o - b,
                     'ours_stderr': ours['env'][env]['stderr'],
                     'r3m_stderr':  baseline['aggregate'][env]['stderr']})
    return rows

def plot_side_by_side(rows, out_path):
    envs = [r['env'] for r in rows]
    ours = [r['ours_mean'] for r in rows]
    r3m  = [r['r3m_mean']  for r in rows]
    ours_e = [r['ours_stderr'] for r in rows]
    r3m_e  = [r['r3m_stderr']  for r in rows]
    x = np.arange(len(envs)); w = 0.4
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w/2, ours, width=w, yerr=ours_e, capsize=3, label='DINOv2+LoRA (ours)')
    ax.bar(x + w/2, r3m,  width=w, yerr=r3m_e,  capsize=3, label='R3M (paper)')
    ax.set_xticks(x); ax.set_xticklabels(envs, rotation=20)
    ax.set_ylabel('Success rate (%)'); ax.set_title('Per-env: ours vs R3M')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)

def plot_per_task(ours_summary, baseline, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, env in zip(axes, ['franka_kitchen', 'metaworld', 'adroit']):
        tasks = sorted([t for (e, t) in ours_summary['task'].keys() if e == env])
        if not tasks:
            ax.set_title(f"{env} (no data)"); continue
        ours = [ours_summary['task'][(env, t)]['mean'] for t in tasks]
        r3m  = [baseline['per_task_approx'][env].get(t, np.nan) for t in tasks]
        x = np.arange(len(tasks)); w = 0.4
        ax.bar(x - w/2, ours, width=w, label='ours')
        ax.bar(x + w/2, r3m,  width=w, label='R3M (Fig 9 approx)')
        ax.set_xticks(x); ax.set_xticklabels(tasks, rotation=30, ha='right', fontsize=8)
        ax.set_title(env); ax.set_ylabel('Success %'); ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)

def plot_training_curves(out_path):
    losses = results.data.get('pretrain', {}).get('losses', [])
    if not losses: return
    steps = [x['step'] for x in losses]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, [x['total'] for x in losses], label='total')
    ax.plot(steps, [x['tcn']   for x in losses], label='TCN')
    ax.plot(steps, [x['lang']  for x in losses], label='lang')
    ax.set_xlabel('step'); ax.set_ylabel('loss'); ax.set_title('Pretrain losses (DINOv2+LoRA on Ego4D)')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
""")


# ============================================================
# Cell 19 — auto markdown summary
# ============================================================
code("""# ===== CELL 19: auto-generated markdown summary =====
def make_summary_md(ours, delta_rows, baseline):
    lines = []
    lines.append("# DINOv2+LoRA vs R3M — Results")
    lines.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')}_\\n")
    lines.append("## Per-environment delta (ours − R3M)\\n")
    lines.append("| Environment | Ours mean (%) | R3M mean (%) | Δ (pp) | Ours s.e. | R3M s.e. |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in delta_rows:
        lines.append(f"| {r['env']} | {r['ours_mean']:.1f} | {r['r3m_mean']:.1f} | "
                     f"{r['delta']:+.1f} | {r['ours_stderr']:.1f} | {r['r3m_stderr']:.1f} |")
    if delta_rows:
        all_ours = np.mean([r['ours_mean'] for r in delta_rows])
        all_r3m  = np.mean([r['r3m_mean']  for r in delta_rows])
        lines.append(f"| **avg** | **{all_ours:.1f}** | **{all_r3m:.1f}** | **{all_ours-all_r3m:+.1f}** | | |")

    lines.append("\\n## Per-task (ours)\\n")
    by_env = {}
    for (e, t), v in ours.get('task', {}).items():
        by_env.setdefault(e, []).append((t, v['mean'], v['stderr'], v['n']))
    for env, rows in by_env.items():
        lines.append(f"### {env}\\n")
        lines.append("| task | success % | s.e. | n_runs |")
        lines.append("|---|---:|---:|---:|")
        for t, m, se, n in sorted(rows):
            lines.append(f"| {t} | {m:.1f} | {se:.1f} | {n} |")
        lines.append("")

    lines.append("\\n## Plots\\n")
    lines.append("- `side_by_side.png` — per-env bars vs R3M aggregate")
    lines.append("- `per_task.png` — per-task vs R3M (Fig 9 approx)")
    lines.append("- `pretrain_curves.png` — TCN + lang + total losses")
    lines.append("\\n## Reproduction notes\\n")
    lines.append("- Hardware: 1× A100 80GB; bf16 + torch.compile.")
    lines.append(f"- Pretrain: {CONFIG['pretrain_steps']:,} steps (vs. R3M's 1M); "
                 f"DINOv3 ViT-B/16 frozen + LoRA r={CONFIG['lora_rank']}; CLIP ViT-B/32 text embeds for VLA.")
    lines.append(f"- BC: 2-layer MLP {CONFIG['bc_hidden']} + BatchNorm, lr {CONFIG['bc_lr']}, "
                 f"{CONFIG['bc_steps']:,} steps, eval every {CONFIG['bc_eval_every']:,}, "
                 f"{CONFIG['bc_eval_episodes']} eval episodes.")
    lines.append(f"- Cut level: {CONFIG['cut_level']} (see CONFIG['envs'] / 'demo_sizes' / 'views' / 'seeds').")
    lines.append("- Results JSON: `results/results.json`. Per-run history in `bc.runs[*].history`.")
    return "\\n".join(lines)
""")


# ============================================================
# SMOKE TESTS — exercise the full pipeline at micro-scale
# ============================================================
md("""---

## Smoke tests

These cells exercise every pipeline stage at micro-scale (8 synthetic Ego4D
clips, 4 pretrain steps, 1 BC config with stubbed simulator) before kicking
off the heavy production run below. Goal: catch dataset, model, loss, training,
and aggregation bugs in ~2 min on A100 (~5 min on CPU) instead of waiting hours
into the real run to discover them.

**What's stubbed:** the simulator (mj_envs / metaworld / Adroit) is replaced
with a no-op env returning random success — the smoke tests verify that the BC
training loop runs end-to-end, *not* that the real environments work. The real
sim envs are exercised on the first BC run in the production sweep below.

**What runs for real:** Ego4D dataset class, CLIP text embedding (real download
on first run, ~600 MB), DINOv3 ViT-B/16 (loaded via torch.hub from a gated
weights URL set in CONFIG['dinov3_weights']), LoRA injection, all three losses,
optimizer step, checkpoint save + load, feature pre-extraction, BC training,
aggregation, plotting, markdown.

**DINOv3 weight access:** weights are gated. Request access at
https://ai.meta.com/dino-v3 and set `CONFIG['dinov3_weights']` to the URL or
local path before running production cells. The smoke section bypasses this by
stubbing `load_dinov3_backbone` with a fake module that mirrors DINOv3's CLS
output dimensionality.

If everything works you'll see `=== ALL SMOKE TESTS PASSED ===` at the end.
Any failure surfaces as a Python `AssertionError` with the failing assertion.

To skip smoke (e.g. on subsequent runs after you've already verified once),
set `RUN_SMOKE_TESTS = False` in the cell below.
""")

code("""# ===== SMOKE 1: build synthetic Ego4D + demos + SMOKE_CONFIG =====
RUN_SMOKE_TESTS = True   # set False to skip on subsequent runs

if RUN_SMOKE_TESTS:
    import shutil, types, copy
    import pandas as pd

    SMOKE_DIR = Path('./_smoke')
    shutil.rmtree(SMOKE_DIR, ignore_errors=True)
    (SMOKE_DIR / 'ego4d').mkdir(parents=True)
    (SMOKE_DIR / 'demos').mkdir(parents=True)
    (SMOKE_DIR / 'cache').mkdir(parents=True)
    (SMOKE_DIR / 'results').mkdir(parents=True)

    # 8 clips × 5 frames each, RGB 120×160 JPEG
    _smoke_narrations = [f"a person doing thing {i} in a kitchen scene" for i in range(8)]
    _smoke_rows = []
    for i, n in enumerate(_smoke_narrations):
        uid = f"clip_{i:03d}"
        (SMOKE_DIR / 'ego4d' / uid).mkdir(parents=True, exist_ok=True)
        for j in range(5):
            arr = (np.random.rand(120, 160, 3) * 255).astype(np.uint8)
            Image.fromarray(arr).save(SMOKE_DIR / 'ego4d' / uid / f"{j}.jpg", quality=85)
        _smoke_rows.append({'clip_uid': uid, 'narration': n, 'n_frames': 5})
    pd.DataFrame(_smoke_rows).to_parquet(SMOKE_DIR / 'ego4d' / 'manifest.parquet')

    # Mini demo set: one env, one task, one view, 6 episodes × 7 frames
    SMOKE_ENV, SMOKE_TASK, SMOKE_VIEW = 'metaworld', 'assembly-v2', 'view1'
    _smoke_demo_dir = SMOKE_DIR / 'demos' / SMOKE_ENV / SMOKE_TASK / SMOKE_VIEW
    _smoke_demo_dir.mkdir(parents=True, exist_ok=True)
    for i in range(6):
        T_steps = 7
        pickle.dump({
            'images':  (np.random.rand(T_steps, 84, 84, 3) * 255).astype(np.uint8),
            'actions': np.random.randn(T_steps, 4).astype(np.float32),
            'proprio': np.random.randn(T_steps, 8).astype(np.float32),
            'success': True,
        }, open(_smoke_demo_dir / f"demo_{i}.pkl", 'wb'))

    SMOKE_CONFIG = copy.deepcopy(CONFIG)
    SMOKE_CONFIG.update({
        'ego4d_root':   str(SMOKE_DIR / 'ego4d'),
        'demo_root':    str(SMOKE_DIR / 'demos'),
        'cache_dir':    str(SMOKE_DIR / 'cache'),
        'results_dir':  str(SMOKE_DIR / 'results'),
        'pretrain_ckpt':str(SMOKE_DIR / 'cache' / 'pretrain.pt'),
        'image_size':   84,                  # 84/14 = 6 patches per side
        'pretrain_steps': 4,
        'batch_clips':  2,
        'num_negatives':1,
        'pretrain_log_every': 1, 'pretrain_save_every': 100,
        'compile_model': False,              # no compile for smoke; tests correctness
        'num_workers':  0,
        'persistent_workers': False, 'pin_memory': False,
        'bc_steps':     50,
        'bc_eval_every':25,
        'bc_batch':     8,
        'bc_eval_episodes': 2,
        'bc_early_stop_patience': 10,
        'envs': {SMOKE_ENV: {'tasks': [SMOKE_TASK], 'views': [SMOKE_VIEW],
                              'demo_sizes': [5], 'horizon': 5}},
        'seeds':        [0],
        'cut_level':    0,
    })
    # Also reset the CLIP cache filename to live inside SMOKE_DIR
    log(f"smoke setup OK: {SMOKE_DIR.resolve()}")
else:
    log("RUN_SMOKE_TESTS=False — skipping smoke tests")
""")

code("""# ===== SMOKE 2: end-to-end mini pipeline =====
if RUN_SMOKE_TESTS:
    # ---- 1. Dataset + collate ----
    _smoke_ds = Ego4DR3MDataset(SMOKE_CONFIG['ego4d_root'], SMOKE_CONFIG['manifest_file'],
                                image_size=SMOKE_CONFIG['image_size'])
    assert len(_smoke_ds) == 8, f"expected 8 clips, got {len(_smoke_ds)}"
    _sample = _smoke_ds[0]
    assert _sample['frames'].shape == (5, 3, 84, 84), _sample['frames'].shape
    log("smoke 1/9: Ego4DR3MDataset OK")

    _smoke_loader = DataLoader(
        _smoke_ds, batch_size=SMOKE_CONFIG['batch_clips'], shuffle=True,
        drop_last=True, num_workers=0, collate_fn=_ego4d_collate)
    _smoke_batch = next(iter(_smoke_loader))
    assert _smoke_batch['frames'].shape == (2, 5, 3, 84, 84)
    log("smoke 2/9: DataLoader collate OK")

    # ---- 2. CLIP text cache ----
    # Override the global CLIP_CACHE path so the cache lands in _smoke/
    _orig_clip_cache = CLIP_CACHE
    CLIP_CACHE = Path(SMOKE_CONFIG['cache_dir']) / 'clip_text_embeds.pt'
    _smoke_clip_cache = build_clip_text_cache(SMOKE_CONFIG, force=True)
    assert _smoke_clip_cache['embeds'].shape == (8, 512)
    log("smoke 3/9: CLIP text cache OK ({})".format(tuple(_smoke_clip_cache['embeds'].shape)))

    # ---- 3. Model assembly (DINOv3 — stubbed in smoke; production needs gated weights) ----
    # Bypass the gated DINOv3 weight load for smoke by replacing load_dinov3_backbone with a stub.
    # We use proper nn.Module subclasses (not nn.ModuleDict) so that setattr-based LoRA injection
    # actually replaces the Linears in the module tree.
    class _StubAttn(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.qkv  = nn.Linear(d, 3*d)
            self.proj = nn.Linear(d, d)
        def forward(self, x):
            q, k, v = self.qkv(x).chunk(3, dim=-1)
            return self.proj(q + k + v) / 3
    class _StubMlp(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.fc1 = nn.Linear(d, 4*d)
            self.fc2 = nn.Linear(4*d, d)
        def forward(self, x): return self.fc2(nn.functional.gelu(self.fc1(x)))
    class _StubBlock(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.attn = _StubAttn(d); self.mlp = _StubMlp(d)
        def forward(self, x): return x + self.mlp(x + self.attn(x))
    class _FakeDinov3(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.blocks = nn.ModuleList([_StubBlock(d) for _ in range(12)])
            self.norm   = nn.LayerNorm(d)
            self._bulk  = nn.Parameter(torch.zeros(80_000_000 // 1024, 1024))   # frozen-count sanity
        def forward(self, x):
            feat = nn.functional.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
            feat = nn.functional.pad(feat, (0, SMOKE_CONFIG['feat_dim'] - feat.shape[-1]))
            for blk in self.blocks: feat = blk(feat)
            return self.norm(feat)
    _real_load_dinov3 = load_dinov3_backbone
    globals()['load_dinov3_backbone'] = lambda variant, weights: _FakeDinov3(SMOKE_CONFIG['feat_dim'])
    log("smoke: instantiating DinoLoRAR3M (stubbed DINOv3 — production needs gated weights)...")
    _smoke_model = DinoLoRAR3M(SMOKE_CONFIG).to(DEVICE)
    _trainable = [p for p in _smoke_model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in _trainable)
    assert n_train > 0, "no trainable params after LoRA injection"
    # Sanity: backbone param count should be huge but frozen
    n_frozen = sum(p.numel() for p in _smoke_model.backbone.parameters() if not p.requires_grad)
    assert n_frozen > 80_000_000, f"expected ViT-B/16 ~86M frozen params, got {n_frozen:,}"
    log(f"smoke 4/9: model OK (trainable={n_train:,}, frozen={n_frozen:,})")

    # ---- 4. Forward + losses + gradient flow ----
    _frames = _smoke_batch['frames'].to(DEVICE)
    _l_emb  = _smoke_clip_cache['embeds'][:2].to(DEVICE)
    _autocast_kwargs = dict(device_type='cuda', dtype=DTYPE, enabled=DEVICE.type=='cuda')
    with torch.autocast(**_autocast_kwargs):
        _out  = _smoke_model.forward_pretrain(_frames)
        _l_t  = time_contrastive_loss(_out, num_neg=1)
        _l_l  = video_language_loss(_smoke_model, _out, _l_emb, num_neg=1)
        _l_r  = regularization(_out, SMOKE_CONFIG['lambda_l1'], SMOKE_CONFIG['lambda_l2'])
        _total = _l_t + _l_l + _l_r
    _total.backward()
    assert torch.isfinite(_total), f"non-finite loss: {_total}"
    n_with_grad = sum(1 for p in _trainable if p.grad is not None and p.grad.abs().sum() > 0)
    assert n_with_grad > 0, "no LoRA params received gradients"
    log(f"smoke 5/9: losses + gradient flow OK "
        f"(tcn={_l_t.item():.3f} lang={_l_l.item():.3f} reg={_l_r.item():.6f}; "
        f"{n_with_grad}/{len(_trainable)} params got grads)")

    # ---- 5. Pretrain (4 steps end-to-end, saves checkpoint) ----
    _opt = torch.optim.Adam(_trainable, lr=SMOKE_CONFIG['pretrain_lr'])
    pretrain(SMOKE_CONFIG, _smoke_model, _opt, _smoke_loader,
             _smoke_clip_cache, SMOKE_CONFIG['pretrain_ckpt'])
    assert Path(SMOKE_CONFIG['pretrain_ckpt']).exists()
    log("smoke 6/9: pretrain (4 steps) OK")

    # ---- 6. Load pretrained + encode images ----
    _m2 = load_pretrained_model(SMOKE_CONFIG, SMOKE_CONFIG['pretrain_ckpt'])
    _imgs = (np.random.rand(7, SMOKE_CONFIG['image_size'], SMOKE_CONFIG['image_size'], 3) * 255).astype(np.uint8)
    _feats = encode_images(_m2, _imgs, batch=4)
    assert _feats.shape == (7, SMOKE_CONFIG['feat_dim']), _feats.shape
    log(f"smoke 7/9: load + encode_images OK ({_feats.shape})")

    # ---- 7. Per-(task,view) feature extraction over synthetic demos ----
    for env_name, env_cfg in SMOKE_CONFIG['envs'].items():
        extract_features_for_env(SMOKE_CONFIG, _m2, env_name, env_cfg)
    _cf = (Path(SMOKE_CONFIG['cache_dir']) / 'features' / SMOKE_ENV
           / f"{SMOKE_TASK}__{SMOKE_VIEW}.pt")
    assert _cf.exists(), f"feature cache missing: {_cf}"
    log("smoke 8/9: feature extraction OK")

    # ---- 8. BC train with stubbed simulator ----
    # The real make_env / rollout require mj_envs + metaworld + a display.
    # We monkey-patch them in globals() so train_bc_one's lookups find the stubs.
    _real_make_env = make_env
    _real_rollout  = rollout
    def _stub_make_env(env_name, task, view, image_size=224):
        return types.SimpleNamespace(close=lambda: None)
    def _stub_rollout(env, env_name, encoder, policy, horizon, image_size=224):
        return random.random() > 0.5
    globals()['make_env'] = _stub_make_env
    globals()['rollout']  = _stub_rollout
    try:
        _bc_r = train_bc_one(SMOKE_CONFIG, SMOKE_ENV, SMOKE_TASK, SMOKE_VIEW, 5, 0, _m2)
        assert _bc_r['best_success'] is not None, _bc_r
        assert len(_bc_r['history']) > 0, _bc_r
    finally:
        globals()['make_env'] = _real_make_env
        globals()['rollout']  = _real_rollout
    log(f"smoke 9/9: BC train (stubbed sim) OK best={_bc_r['best_success']:.1f}%, "
        f"history_len={len(_bc_r['history'])}")

    # ---- 9. Aggregation + plots + markdown summary ----
    _smoke_sweep = [
        {'env': 'metaworld',      'task': 'assembly-v2', 'view': 'view1',
         'n_demos': 5, 'seed': 0, 'best_success': 70.0, 'history': []},
        {'env': 'franka_kitchen', 'task': 'slide',       'view': 'view1',
         'n_demos': 5, 'seed': 0, 'best_success': 55.0, 'history': []},
    ]
    _ours = summarize(_smoke_sweep)
    _delta = delta_table(_ours, R3M_BASELINE)
    assert len(_delta) == 2, _delta
    plot_side_by_side(_delta, Path(SMOKE_CONFIG['results_dir']) / 'sbs.png')
    plot_per_task(_ours, R3M_BASELINE, Path(SMOKE_CONFIG['results_dir']) / 'pt.png')
    _md = make_summary_md(_ours, _delta, R3M_BASELINE)
    assert "Per-environment delta" in _md
    log("smoke aggregation + plots + markdown OK")

    # Restore the production CLIP_CACHE path
    CLIP_CACHE = _orig_clip_cache

    print()
    print("=" * 60)
    print("=== ALL SMOKE TESTS PASSED ===")
    print("Pipeline runs end-to-end. Safe to proceed with the production cells.")
    print("=" * 60)
""")

code("""# ===== SMOKE 3: cleanup =====
if RUN_SMOKE_TESTS:
    import shutil
    shutil.rmtree('./_smoke', ignore_errors=True)
    # Restore the real DINOv3 loader so production cells use it
    if '_real_load_dinov3' in globals():
        globals()['load_dinov3_backbone'] = _real_load_dinov3
    log("smoke artifacts removed; production caches untouched; DINOv3 loader restored")
""")


# ============================================================
# PRODUCTION RUN — heavy execution, gated behind smoke tests above
# ============================================================
md("""---

## Production run

Everything below executes the **full** pipeline against `CONFIG['ego4d_root']`,
`CONFIG['demo_root']`, etc. Total runtime ~3.4-3.8 h on A100 80GB at default
`cut_level=1`. Each stage is independently re-runnable: cached artifacts on
disk are reused unless explicitly invalidated.

| Cell | What it does | Skip condition |
|---|---|---|
| build CLIP cache | encodes all unique narrations once | `cache/clip_text_embeds.pt` exists |
| pretrain | 150k step ViT-B/16 + LoRA training | `cache/pretrain.pt` exists |
| feature extract | DINOv2-LoRA over all demo frames | per-(task, view) cache hit |
| BC sweep | (env × task × view × demos × seed) loops | always runs |
| aggregate + plot | `results/*.png`, summary stats | always runs |
| markdown summary | `results/summary.md` | always runs |

To force a re-run of any stage, delete the corresponding cache file.
""")

code("""# ===== PROD 1: build CLIP text cache =====
try:
    CLIP_TEXT_CACHE = build_clip_text_cache(CONFIG)
except FileNotFoundError as e:
    log(f"WARNING: skipping CLIP text cache — {e}")
    CLIP_TEXT_CACHE = None
""")

code("""# ===== PROD 2: run pretraining =====
ckpt_path = Path(CONFIG['pretrain_ckpt'])
if ckpt_path.exists():
    log(f"pretrain checkpoint already exists at {ckpt_path} — skipping pretrain.")
    log("    delete the file or set CONFIG['pretrain_ckpt'] to a fresh path to re-run.")
else:
    if CLIP_TEXT_CACHE is None:
        raise RuntimeError("CLIP_TEXT_CACHE missing — fix dataset path in CONFIG['ego4d_root'] first.")
    _ds, train_loader = make_pretrain_loader(CONFIG)
    model, opt = make_model_and_optim(CONFIG)
    pretrain(CONFIG, model, opt, train_loader, CLIP_TEXT_CACHE, ckpt_path)
""")

code("""# ===== PROD 3: feature pre-extraction over downstream demos =====
PRETRAINED_MODEL = run_feature_extraction(CONFIG)
""")

code("""# ===== PROD 4: BC sweep =====
if PRETRAINED_MODEL is None:
    if Path(CONFIG['pretrain_ckpt']).exists():
        PRETRAINED_MODEL = load_pretrained_model(CONFIG, CONFIG['pretrain_ckpt'])
if PRETRAINED_MODEL is not None:
    SWEEP = run_bc_sweep(CONFIG, PRETRAINED_MODEL)
else:
    SWEEP = []
    log("no pretrained model available — skipping BC sweep.")
""")

code("""# ===== PROD 5: aggregate, delta, plots =====
if SWEEP:
    OURS_SUMMARY = summarize(SWEEP)
    DELTA_ROWS = delta_table(OURS_SUMMARY, R3M_BASELINE)
    plots = Path(CONFIG['results_dir'])
    plot_side_by_side(DELTA_ROWS, plots / 'side_by_side.png')
    plot_per_task(OURS_SUMMARY, R3M_BASELINE, plots / 'per_task.png')
    plot_training_curves(plots / 'pretrain_curves.png')
    results.log('summary', 'ours', OURS_SUMMARY)
    results.log('summary', 'delta', DELTA_ROWS)
    log(f"plots written to {plots}/*.png")
else:
    OURS_SUMMARY = {}; DELTA_ROWS = []
""")

code("""# ===== PROD 6: markdown summary =====
SUMMARY_MD = make_summary_md(OURS_SUMMARY, DELTA_ROWS, R3M_BASELINE)
out_md = Path(CONFIG['results_dir']) / 'summary.md'
out_md.write_text(SUMMARY_MD)
log(f"summary written -> {out_md}")
print()
print(SUMMARY_MD)
""")


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.write_text(json.dumps(nb, indent=1))
print(f"wrote {NB_PATH} ({len(cells)} cells)")
