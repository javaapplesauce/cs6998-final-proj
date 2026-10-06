"""Build baseline_3_dinov2_sam2.ipynb — DINOv2 + SAM 2 frozen, no Ego4D pretraining."""
import json, uuid
from pathlib import Path

NB_PATH = Path(__file__).parent / "baseline_3_dinov2_sam2.ipynb"
cells = []

def _id(): return uuid.uuid4().hex[:12]
def md(src):
    cells.append({"cell_type": "markdown", "id": _id(), "metadata": {},
                  "source": src.splitlines(keepends=True)})
def code(src):
    cells.append({"cell_type": "code", "id": _id(), "execution_count": None,
                  "metadata": {}, "outputs": [],
                  "source": src.splitlines(keepends=True)})


# ============================================================
# Cell 1 — title & 3-baseline framing
# ============================================================
md("""# Baseline 3: DINOv2 + SAM 2 (frozen) on Robosuite Lift

Frozen-foundation-model approach: no Ego4D pretraining, no LoRA. DINOv2 ViT-B/14
provides dense spatial features; SAM 2 (single center-point prompt) provides a
foreground mask that filters the feature map element-wise before global average
pooling. Only a small BC MLP head is trainable.

**Pragmatic deviation from the proposal:** the original proposal called for
DINOv3 + SAM 3 with text-prompted concept segmentation. Both are gated. To fit
the remaining time budget we use the public counterparts:
- **DINOv2** instead of DINOv3 (12× less pretrain data but publicly available)
- **SAM 2** with a single center-point prompt instead of SAM 3's text grounding
  (loses open-vocabulary grounding but the Lift cube is always near the image
  center, so a centered point segments the target reliably)

## Comparison framing

- **Baseline 1** — R3M (Nair et al., 2022): paper numbers from Tables 1-3 (kept
  as context; not directly comparable since R3M never reported on Robosuite Lift).
- **Baseline 2** (`baseline_2_dinov3_lora.ipynb`): DINOv3 ViT-B/16 + LoRA, R3M's
  losses on Ego4D. Aspirational — needs Ego4D pretrain that doesn't fit the budget.
- **Baseline 3** (this notebook): DINOv2 ViT-B/14 + SAM 2, fully frozen, **no
  Ego4D pretraining**, BC MLP only, on Robosuite Lift with 50 teleop demos.

**Hardware target:** 1× A100 80GB or Colab T4/A100, ≤ 4 h wall clock.
""")


# ============================================================
# Cell 2 — bottleneck analysis
# ============================================================
md("""## Bottleneck analysis & wall-clock budget (single task: Lift, 50 demos)

| Stage | Configuration | Estimate |
|---|---|---|
| HDF5 → pickle conversion | 50 demos × ~83 frames, transpose CHW→HWC | ~2 min |
| Demo feature pre-extraction (DINOv2 dense map) | 1 task-view × ~4000 frames × ~10 ms | ~1 min |
| Demo mask pre-extraction (SAM 2, centered point) | 1 task-view × ~4000 frames × ~80 ms | ~5-8 min |
| BC sweep | 1 task × 1 view × 3 sizes × 3 seeds = 9 runs | ~30-60 min |
| Aggregate + plots + summary | | ~1 min |
| **Total** | | **~40-75 min** |

**Why no pretrain stage:** both DINOv2 and SAM 2 are loaded already-trained from foundation-model weights. We only optimize a 3-layer MLP policy head (~300K params) on top.

**Why one center-point prompt to SAM 2:** the Lift cube is at the workspace center in every frame, so a single fixed prompt at `(image_size//2, image_size//2)` reliably segments it. No need for text grounding on this single task. The mask is the SAM 2 output binarized; multiple-mask outputs are unioned.
""")


# ============================================================
# Cell 3 — environment setup
# ============================================================
md("""## Environment setup

```bash
# Core (already in baseline 2's stack)
pip install "torch>=2.3" torchvision pillow numpy pandas h5py matplotlib

# SAM 2 (public, no gating)
git clone https://github.com/facebookresearch/sam2.git
cd sam2 && pip install -e . && cd ..
mkdir -p checkpoints && wget -P checkpoints \
    https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

# Robosuite for the Lift env
pip install robosuite

# DINOv2 is loaded via torch.hub at runtime — no manual download needed.
```

The smoke section stubs both DINOv2 and SAM 2 — you can run smoke without weight access. The HDF5 conversion cell also runs offline against the actual file.
""")


# ============================================================
# Cell 4 — CONFIG + imports + setup
# ============================================================
code("""# ===== CELL 4: CONFIG + imports + setup =====
import os, sys, json, time, math, random, hashlib, pickle, shutil, copy
from pathlib import Path

CONFIG = {
    # ---- paths ----
    'hdf5_src':      '/content/cavr/data/demos_Lift/demos.hdf5',   # the user's teleop demo cache
    'demo_root':     '/content/cavr/data/r3m_demos',               # where HDF5→pickle conversion writes
    'cache_dir':     './cache_b3',
    'results_dir':   './results_b3',

    # ---- model ----
    'dinov2_variant':   'dinov2_vitb14',           # ViT-B/14, public, no gating
    'feat_dim':         768,                       # DINOv2 ViT-B/14 output dim
    'sam2_config':     'sam2_hiera_l.yaml',        # SAM 2 config name (large variant)
    'sam2_ckpt':       './checkpoints/sam2_hiera_large.pt',
    'image_size':       518,                       # match Lift demo collection resolution
    'feature_res':      37,                        # 518/14 = 37 patches per side
    'mask_threshold':   0.5,
    'sam2_point':       None,                      # None = use image center; or set (x, y)

    # ---- BC eval (R3M protocol, Appendix B.5; preserved for comparability across baselines) ----
    'bc_lr':            1e-3,
    'bc_batch':         32,
    'bc_steps':         20_000,
    'bc_eval_every':    1_000,
    'bc_hidden':        (256, 256, 256),           # 3-layer MLP per proposal §3.1
    'bc_use_layernorm': True,
    'bc_eval_episodes': 10,
    'bc_early_stop_patience': 3,

    # ---- SAM 2 eval policy ----
    'sam_refresh_steps': 0,                        # 0 = run once per episode; >0 = re-run every N steps

    # ---- envs: single-task, Lift-only ----
    'envs': {
        'robosuite': {
            'tasks': ['Lift'],
            'views': ['view1'],
            'demo_sizes': [10, 25, 50],            # subsets of the 50 collected
            'horizon': 100,                        # Lift default
        },
    },

    'seeds': [0, 1, 2],
    'cut_level': 0,                                # no cuts; single-task pipeline is already small

    # ---- compute ----
    'precision':     'bf16',
    'compile_model': False,
    'num_workers':   2,
    'pin_memory':    True,
}

def apply_cuts(cfg):
    cl = cfg['cut_level']
    if cl >= 2:
        for e in cfg['envs'].values():
            if 'demo_sizes' in e and len(e['demo_sizes']) >= 3:
                e['demo_sizes'] = [e['demo_sizes'][0], e['demo_sizes'][-1]]
    if cl >= 3: cfg['seeds'] = cfg['seeds'][:2]
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

print(f"[setup] device={DEVICE}, dtype={DTYPE}, "
      f"envs={list(CONFIG['envs'].keys())}, dinov2={CONFIG['dinov2_variant']}")
""")


# ============================================================
# Cell 5/6 — Colab setup (reuses pattern from baseline 2)
# ============================================================
md("""---

## Optional: Colab / Drive setup

Mirrors baseline 2's Colab cell — mounts Drive, clones repo, checks deps, patches `CONFIG` paths so caches persist on Drive across Colab disconnects. No-op outside Colab when `DRIVE_ROOT` is inaccessible.
""")

code("""# ===== Colab setup: mount Drive, clone repo, check deps, patch CONFIG paths =====
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

REPO_DIR = Path('/content/cs6998-final-proj') if IN_COLAB else Path.cwd()
if IN_COLAB and not REPO_DIR.exists():
    subprocess.run(['git', 'clone',
                    'https://github.com/javaapplesauce/cs6998-final-proj',
                    str(REPO_DIR)], check=True)
sys.path.insert(0, str(REPO_DIR))

REQUIRED = {'torch':'torch>=2.3', 'torchvision':'torchvision', 'PIL':'pillow',
            'numpy':'numpy', 'pandas':'pandas', 'matplotlib':'matplotlib',
            'h5py':'h5py'}
missing = [pkg for mod, pkg in REQUIRED.items() if importlib.util.find_spec(mod) is None]
if missing:
    print(f"[warn] missing: pip install {' '.join(missing)}")
else:
    print("[ok] core packages present")
for opt in ('sam2', 'robosuite'):
    if importlib.util.find_spec(opt) is None:
        print(f"[warn] {opt} not importable — required for production; smoke section stubs it")
    else:
        print(f"[ok] {opt} importable")

# In Colab we don't need to re-patch CONFIG['hdf5_src'] — it's already /content/cavr/...
# We do patch cache_dir / results_dir to Drive so they survive disconnects.
if IN_COLAB:
    DRIVE_RUNS = DRIVE_ROOT / 'cs6998-final-proj-runs'
    CONFIG.update({
        'cache_dir':    str(DRIVE_RUNS / 'cache_b3'),
        'results_dir':  str(DRIVE_RUNS / 'results_b3'),
    })
    Path(CONFIG['cache_dir']).mkdir(parents=True, exist_ok=True)
    Path(CONFIG['results_dir']).mkdir(parents=True, exist_ok=True)
    print(f"\\n[ok] caches on Drive: {CONFIG['cache_dir']}")
else:
    print(f"\\n[info] non-Colab — leaving Cell 4 defaults intact.")
""")


# ============================================================
# Cell 7 — logger (same as baseline 2)
# ============================================================
code("""# ===== CELL 7: dual logger (stdout + results.json) =====
RESULTS_PATH = Path(CONFIG['results_dir']) / 'results.json'

class Results:
    def __init__(self, path):
        self.path = Path(path)
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        else:
            self.data = {'baseline_id': 'b3_dinov2_sam2', 'config': CONFIG,
                         'bc': {}, 'meta': {'started_at': time.time()}}
            self._flush()
    def log(self, section, key, value):
        self.data.setdefault(section, {})[key] = value; self._flush()
    def append(self, section, key, value):
        self.data.setdefault(section, {}).setdefault(key, []).append(value); self._flush()
    def _flush(self):
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.data, indent=2, default=str)); tmp.replace(self.path)

results = Results(RESULTS_PATH)
def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
log(f"results.json -> {RESULTS_PATH}")
""")


# ============================================================
# Cell 8 — R3M baseline (hardcoded, same as baseline 2)
# ============================================================
code("""# ===== CELL 8: R3M baseline (Tables 1-3) =====
R3M_BASELINE = {
    'aggregate': {
        'franka_kitchen': {'mean': 53.1, 'stderr': 2.7},
        'metaworld':      {'mean': 69.2, 'stderr': 2.0},
        'adroit':         {'mean': 65.0, 'stderr': 1.7},
        'all_domains':    {'mean': 62.4, 'stderr': 1.3},
    },
    'per_task_approx': {
        'franka_kitchen': {'slide': 60, 'light': 50, 'door': 55, 'knob': 50, 'micro': 50},
        'metaworld':      {'assembly-v2': 65, 'bin-picking-v2': 70,
                           'button-press-topdown-v2': 80, 'drawer-open-v2': 90, 'hammer-v2': 60},
        'adroit':         {'pen-v0': 60, 'relocate-v0': 70},
    },
}
results.log('baseline', 'r3m', R3M_BASELINE)
log("R3M baseline table loaded")
""")


# ============================================================
# Cell 9 — task → text prompt map (for SAM 3)
# ============================================================
code("""# ===== CELL 9: HDF5 -> pickle demo conversion =====
# The teleop collection writes a single HDF5 file with 50 demos at:
#   /content/cavr/data/demos_Lift/demos.hdf5
# Structure:
#   demo_<i>/actions: (T, 6) float32          # 6-DOF EE delta (3 pos + 3 rot)
#   demo_<i>/images : (T, 3, 518, 518) uint8  # CHW layout
#   demo_<i>/proprio: (T, 14) float32         # 7 joint + 7 EE pose
#   attrs: env_name='Lift', action_dim=6, proprio_dim=14, image_size=518
#
# We convert to the pickle layout the rest of the notebook consumes:
#   <demo_root>/robosuite/Lift/view1/demo_<i>.pkl
#   each pickle is dict with images (HWC), actions, proprio, success.

import h5py, pickle, shutil
import numpy as np

def convert_hdf5_to_pickles(hdf5_path, dest_dir, force=False):
    \"\"\"Convert the teleop HDF5 to per-demo pickle files; idempotent unless force=True.\"\"\"
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(dest_dir.glob('demo_*.pkl'))
    if not force and existing:
        log(f"convert: {len(existing)} pickles already at {dest_dir} — skipping (set force=True to redo)")
        return len(existing)

    if force:
        for f in existing: f.unlink()

    with h5py.File(hdf5_path, 'r') as f:
        keys = sorted(f.keys(), key=lambda k: int(k.split('_')[1]))
        for i, key in enumerate(keys):
            g = f[key]
            imgs_chw = np.asarray(g['images'])              # (T, 3, H, W) uint8
            imgs     = imgs_chw.transpose(0, 2, 3, 1)        # (T, H, W, 3)
            pickle.dump({
                'images':  imgs,
                'actions': np.asarray(g['actions']),         # (T, A)
                'proprio': np.asarray(g['proprio']),         # (T, P)
                'success': True,                              # # TODO: confirm collection filters to success
            }, open(dest_dir / f"demo_{i}.pkl", 'wb'))
    log(f"convert: wrote {len(keys)} pickles to {dest_dir}")
    return len(keys)

RUN_CONVERSION = True   # set False if pickles are already in place

if RUN_CONVERSION:
    src = Path(CONFIG['hdf5_src'])
    if not src.exists():
        log(f"[warn] hdf5_src={src} does not exist — set CONFIG['hdf5_src'] correctly or copy the file")
    else:
        dest = Path(CONFIG['demo_root']) / 'robosuite' / 'Lift' / 'view1'
        convert_hdf5_to_pickles(src, dest)
else:
    log("RUN_CONVERSION=False — skipping HDF5 conversion")
""")


# ============================================================
# Cell 10 — DINOv3 dense feature extractor (frozen)
# ============================================================
code("""# ===== CELL 10: DINOv2 dense feature extractor (frozen) =====
# Baseline 3 uses DINOv2's *patch tokens* (h × w × d), not the CLS token —
# spatial structure is preserved so SAM 2's mask can do something meaningful at
# the feature-map resolution. We use get_intermediate_layers(reshape=True) to
# get a (B, D, h, w) tensor.

def load_dinov2_backbone(variant: str):
    return torch.hub.load('facebookresearch/dinov2', variant)

class DinoV2DenseEncoder(nn.Module):
    \"\"\"Frozen DINOv2. Returns (B, D, h, w) feature map.\"\"\"
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = load_dinov2_backbone(cfg['dinov2_variant'])
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    @torch.no_grad()
    def forward(self, x):
        out = self.backbone.get_intermediate_layers(
            x, n=1, reshape=True, return_class_token=False, norm=True
        )
        return out[0]   # (B, D, h, w)

@torch.no_grad()
def encode_image_to_feature_map(model: 'DinoV2DenseEncoder', img_uint8, image_size=224):
    \"\"\"Single image (H, W, 3) uint8 -> (D, h, w) tensor on cpu.\"\"\"
    import numpy as np
    from PIL import Image
    if img_uint8.shape[0] != image_size or img_uint8.shape[1] != image_size:
        img_uint8 = np.asarray(Image.fromarray(img_uint8).resize(
            (image_size, image_size), Image.Resampling.BICUBIC))
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
    x = torch.from_numpy(img_uint8).to(DEVICE).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    x = (x - mean) / std
    with torch.autocast(device_type='cuda', dtype=DTYPE, enabled=DEVICE.type == 'cuda'):
        feat = model(x)
    return feat[0].float().cpu()
""")


# ============================================================
# Cell 11 — SAM 3 concept mask extractor (frozen)
# ============================================================
code("""# ===== CELL 11: SAM 2 mask extractor (frozen, single center-point prompt) =====
# SAM 2 doesn't natively take text — for the Lift task the cube is always near
# the workspace center, so a single fixed point prompt at the image center
# (or CONFIG['sam2_point']) reliably picks out the target.

class Sam2Masker:
    def __init__(self, cfg):
        self.cfg = cfg
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            sam2_model = build_sam2(cfg['sam2_config'], cfg['sam2_ckpt'], device=DEVICE)
            self.predictor = SAM2ImagePredictor(sam2_model)
        except ImportError:
            raise ImportError(
                "sam2 not installed. git clone facebookresearch/sam2 && pip install -e ."
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"SAM 2 checkpoint missing at {cfg['sam2_ckpt']}. Download with:\\n"
                f"  mkdir -p checkpoints && wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
            )

    @torch.no_grad()
    def mask(self, img_uint8, feature_res: int):
        \"\"\"img_uint8: (H, W, 3) -> (feature_res, feature_res) float mask in {0,1}.\"\"\"
        import numpy as np
        H, W = img_uint8.shape[:2]
        pt = self.cfg['sam2_point']
        if pt is None: pt = (W // 2, H // 2)
        self.predictor.set_image(img_uint8)
        masks, scores, _ = self.predictor.predict(
            point_coords=np.array([pt], dtype=np.float32),     # (1, 2)
            point_labels=np.array([1], dtype=np.int32),         # 1 = foreground
            multimask_output=False,
        )
        # masks: (1, H, W) bool; take the single mask.
        union = masks[0].astype('float32')
        m = torch.from_numpy(union).unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)
        m = F.adaptive_avg_pool2d(m, (feature_res, feature_res))[0, 0]
        return (m > self.cfg['mask_threshold']).float()
""")


# ============================================================
# Cell 12 — combined feature extractor (DINOv3 + mask + GAP)
# ============================================================
code("""# ===== CELL 12: combined visual representation: DINOv2 dense ⊙ SAM 2 mask -> GAP =====
def masked_feature_pool(feat_map, mask):
    \"\"\"feat_map: (D, h, w); mask: (h, w) -> z_vis: (D,)
    Empty-mask fallback: unmasked GAP (keeps policy operational if SAM misses).\"\"\"
    if mask.sum() < 1.0:
        return feat_map.mean(dim=(1, 2))
    masked = feat_map * mask.unsqueeze(0)
    return masked.sum(dim=(1, 2)) / mask.sum().clamp_min(1.0)

@torch.no_grad()
def encode_episode_features(dinov2, sam2, images_uint8, cfg):
    \"\"\"Returns per-frame z_vis tensor: (T, D).\"\"\"
    feats = []
    for img in images_uint8:
        feat_map = encode_image_to_feature_map(dinov2, img, cfg['image_size'])  # (D, h, w)
        mask     = sam2.mask(img, cfg['feature_res'])                            # (h, w)
        z = masked_feature_pool(feat_map.to(DEVICE), mask.to(DEVICE)).float().cpu()
        feats.append(z)
    return torch.stack(feats, dim=0)
""")


# ============================================================
# Cell 13 — BC dataset + 3-layer MLP+LayerNorm policy
# ============================================================
code("""# ===== CELL 13: BC dataset + 3-layer MLP + LayerNorm policy =====
class FeatureBCDataset(Dataset):
    def __init__(self, cache_file, n_demos):
        bundle = torch.load(cache_file, map_location='cpu', weights_only=False)
        eps = bundle['episodes'][:n_demos]
        self.feats   = torch.cat([e['features'] for e in eps], dim=0)
        self.proprio = torch.cat([e['proprio']  for e in eps], dim=0)
        self.actions = torch.cat([e['actions']  for e in eps], dim=0)
    def __len__(self): return self.feats.shape[0]
    def __getitem__(self, i):
        return torch.cat([self.feats[i], self.proprio[i]], dim=-1), self.actions[i]

class BCPolicyB3(nn.Module):
    \"\"\"Proposal §3.1: 3-layer MLP with ReLU + LayerNorm.\"\"\"
    def __init__(self, in_dim, act_dim, hidden=(256, 256, 256), use_layernorm=True):
        super().__init__()
        layers = []; d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h)]
            if use_layernorm: layers += [nn.LayerNorm(h)]
            layers += [nn.ReLU()]
            d = h
        layers += [nn.Linear(d, act_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)
""")


# ============================================================
# Cell 14 — sim env wrappers (mirrors baseline 2)
# ============================================================
code("""# ===== CELL 14: Robosuite Lift env wrapper =====
def make_env(env_name, task, view, image_size=518):
    if env_name == 'robosuite' and task == 'Lift':
        import robosuite
        # # TODO: confirm — these settings should match what the demo collection script used.
        # If demos used a different camera or controller config, BC rollouts will see a
        # different observation distribution than training -> the policy can fail silently.
        env = robosuite.make(
            env_name='Lift',
            robots='Panda',
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names='agentview',
            camera_heights=image_size,
            camera_widths=image_size,
            control_freq=20,
            horizon=CONFIG['envs']['robosuite']['horizon'],
        )
        return env
    raise ValueError(f"unknown env/task: {env_name}/{task}")

def render_rgb(env, image_size=518):
    \"\"\"Robosuite returns the rendered image in the obs dict under '<camera>_image'.\"\"\"
    obs = env._get_observations() if hasattr(env, '_get_observations') else env.observation_spec()
    # Robosuite stores images with the camera prefix:
    for k in ('agentview_image', 'frontview_image', 'sideview_image'):
        if k in obs:
            img = obs[k]
            # Robosuite returns BGR uint8 in HWC. Convert to RGB.
            return img[:, :, ::-1].copy()
    raise RuntimeError(f"no camera image found in obs keys: {list(obs.keys())}")

def get_proprio(env, env_name):
    \"\"\"14-dim: matches the demo collection (7 joint pos + 7 EE pose). # TODO: confirm.\"\"\"
    obs = env._get_observations() if hasattr(env, '_get_observations') else env.observation_spec()
    import numpy as np
    qpos = np.asarray(obs.get('robot0_joint_pos', np.zeros(7))).flatten()
    eef_pos = np.asarray(obs.get('robot0_eef_pos', np.zeros(3))).flatten()
    eef_quat = np.asarray(obs.get('robot0_eef_quat', np.zeros(4))).flatten()
    return np.concatenate([qpos, eef_pos, eef_quat])[:14]
""")


# ============================================================
# Cell 15 — rollout (live DINOv3 + SAM at episode start)
# ============================================================
code("""# ===== CELL 15: rollout with frozen DINOv2 + SAM 2 (mask cached per episode) =====
@torch.no_grad()
def rollout(env, env_name, dinov2, sam2, policy, horizon,
            sam_refresh_steps=0, image_size=518):
    \"\"\"
    sam_refresh_steps=0 -> SAM 2 runs once at frame 0, mask cached for the episode.
    sam_refresh_steps=K -> SAM 2 re-runs every K steps.
    \"\"\"
    from PIL import Image
    import numpy as np
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    success = False; cached_mask = None

    for t in range(horizon):
        rgb = render_rgb(env, image_size)
        if rgb.shape[0] != image_size or rgb.shape[1] != image_size:
            rgb = np.asarray(Image.fromarray(rgb).resize((image_size, image_size), Image.Resampling.BICUBIC))

        if cached_mask is None or (sam_refresh_steps > 0 and t % sam_refresh_steps == 0):
            cached_mask = sam2.mask(rgb, CONFIG['feature_res']).to(DEVICE)

        feat_map = encode_image_to_feature_map(dinov2, rgb, image_size).to(DEVICE)
        z_vis    = masked_feature_pool(feat_map, cached_mask)

        p = torch.as_tensor(get_proprio(env, env_name), device=DEVICE, dtype=torch.float32)
        a = policy(torch.cat([z_vis, p], dim=-1).unsqueeze(0)).squeeze(0).cpu().numpy()
        step_out = env.step(a)
        if len(step_out) == 5:
            obs, _, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        else:
            obs, _, done, info = step_out
        # Robosuite Lift returns 'success' in info; check that, plus reward as fallback.
        if isinstance(info, dict) and (info.get('success') or info.get('goal_achieved')):
            success = True
        if done: break
    return success
""")


# ============================================================
# Cell 16 — BC train + plateau early-stop
# ============================================================
code("""# ===== CELL 16: BC train + plateau early-stop =====
def train_bc_one(cfg, env_name, task, view, n_demos, seed, dinov2, sam2):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    cache_file = Path(cfg['cache_dir']) / 'features' / env_name / f"{task}__{view}.pt"
    if not cache_file.exists():
        return {'best_success': None, 'reason': f'features missing for {cache_file}'}

    ds = FeatureBCDataset(cache_file, n_demos=n_demos)
    if len(ds) == 0: return {'best_success': None, 'reason': 'empty dataset'}
    in_dim = ds.feats.shape[1] + ds.proprio.shape[1]
    act_dim = ds.actions.shape[1]
    policy = BCPolicyB3(in_dim, act_dim, hidden=cfg['bc_hidden'],
                         use_layernorm=cfg['bc_use_layernorm']).to(DEVICE)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg['bc_lr'])
    loader = DataLoader(ds, batch_size=cfg['bc_batch'], shuffle=True, drop_last=True,
                        num_workers=0, pin_memory=cfg['pin_memory'])

    horizon = (cfg['envs'][env_name].get('horizons', {}) or {}).get(task,
              cfg['envs'][env_name].get('horizon'))
    env = make_env(env_name, task, view, image_size=cfg['image_size'])

    best = -1.0; plateau = 0; history = []
    iterator = iter(loader)
    for step in range(1, cfg['bc_steps'] + 1):
        try: x, a = next(iterator)
        except StopIteration:
            iterator = iter(loader); x, a = next(iterator)
        x = x.to(DEVICE, non_blocking=True); a = a.to(DEVICE, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(policy(x), a)
        loss.backward(); opt.step()

        if step % cfg['bc_eval_every'] == 0:
            policy.eval()
            successes = sum(rollout(env, env_name, dinov2, sam2, policy, horizon,
                                     sam_refresh_steps=cfg['sam_refresh_steps'],
                                     image_size=cfg['image_size'])
                            for _ in range(cfg['bc_eval_episodes']))
            rate = 100.0 * successes / cfg['bc_eval_episodes']
            history.append({'step': step, 'success_pct': rate, 'loss': float(loss.item())})
            if rate > best: best = rate; plateau = 0
            else: plateau += 1
            log(f"  [{env_name}/{task}/{view}/n{n_demos}/s{seed}] "
                f"step {step} success={rate:.1f}% best={best:.1f}% plateau={plateau}")
            if plateau >= cfg['bc_early_stop_patience']:
                log(f"  early stop at step {step}"); break
            policy.train()
    if hasattr(env, 'close'): env.close()
    return {'best_success': float(best), 'history': history, 'env': env_name,
            'task': task, 'view': view, 'n_demos': n_demos, 'seed': seed}
""")


# ============================================================
# Cell 17 — BC sweep
# ============================================================
code("""# ===== CELL 17: BC sweep over (env, task, view, n_demos, seed) =====
def run_bc_sweep(cfg, dinov2, sam2):
    sweep = []; sweep_t0 = time.time()
    for env_name, env_cfg in cfg['envs'].items():
        for task in env_cfg['tasks']:
            for view in env_cfg['views']:
                for n_demos in env_cfg['demo_sizes']:
                    for seed in cfg['seeds']:
                        key = f"{env_name}/{task}/{view}/n{n_demos}/s{seed}"
                        try:
                            r = train_bc_one(cfg, env_name, task, view, n_demos, seed, dinov2, sam2)
                        except Exception as e:
                            log(f"FAIL {key}: {type(e).__name__}: {e}")
                            r = {'best_success': None, 'error': f"{type(e).__name__}: {e}"}
                        r['key'] = key
                        sweep.append(r)
                        results.append('bc', 'runs', r)
                        log(f"sweep: {len(sweep)} runs done, elapsed={(time.time()-sweep_t0)/60:.1f}min")
    return sweep
""")


# ============================================================
# Cell 18 — pre-extract demo features + masks
# ============================================================
code("""# ===== CELL 18: pre-extract per-frame features + SAM masks for all demo episodes =====
# After this cell: cache/features/<env>/<task>__<view>.pt contains a list of episodes,
# each with 'features' (T, D), 'actions' (T, A), 'proprio' (T, P), 'success' bool.
def extract_features_for_env(cfg, dinov2, sam2, env_name, env_cfg):
    out_dir = Path(cfg['cache_dir']) / 'features' / env_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for task in env_cfg['tasks']:
        for view in env_cfg['views']:
            cache_file = out_dir / f"{task}__{view}.pt"
            if cache_file.exists():
                log(f"features cached: {cache_file}"); continue
            demo_glob = Path(cfg['demo_root']) / env_name / task / view
            if not demo_glob.exists():
                log(f"# TODO: confirm demo path missing: {demo_glob}"); continue
            episodes = []
            for f in sorted(demo_glob.glob('demo_*.pkl')):
                with open(f, 'rb') as fp:
                    d = pickle.load(fp)
                feats = encode_episode_features(dinov2, sam2, d['images'], cfg)
                episodes.append({
                    'features': feats,
                    'actions':  torch.as_tensor(d['actions'], dtype=torch.float32),
                    'proprio':  torch.as_tensor(d['proprio'], dtype=torch.float32),
                    'success':  bool(d.get('success', True)),
                })
            torch.save({'env': env_name, 'task': task, 'view': view, 'episodes': episodes},
                        cache_file)
            log(f"saved {cache_file} ({len(episodes)} demos)")

def run_feature_extraction(cfg):
    log("loading DINOv2 + SAM 2...")
    dinov2 = DinoV2DenseEncoder(cfg).to(DEVICE)
    sam2   = Sam2Masker(cfg)
    for env_name, env_cfg in cfg['envs'].items():
        extract_features_for_env(cfg, dinov2, sam2, env_name, env_cfg)
    return dinov2, sam2
""")


# ============================================================
# Cell 19 — aggregation, delta vs R3M (and optionally vs B2)
# ============================================================
code("""# ===== CELL 19: aggregate + delta vs R3M (optional vs Baseline 2) + plots =====
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def summarize(sweep):
    by_env, by_task, by_view, by_demos = {}, {}, {}, {}
    for r in sweep:
        if r.get('best_success') is None: continue
        by_env.setdefault(r['env'], []).append(r['best_success'])
        by_task.setdefault((r['env'], r['task']), []).append(r['best_success'])
        by_view.setdefault((r['env'], r['view']), []).append(r['best_success'])
        by_demos.setdefault((r['env'], r['n_demos']), []).append(r['best_success'])
    def red(d):
        return {k: {'mean': float(np.mean(v)), 'stderr': float(np.std(v)/np.sqrt(len(v))),
                    'n': len(v)} for k, v in d.items()}
    return {'env': red(by_env), 'task': red(by_task),
            'view': red(by_view), 'demos': red(by_demos)}

def delta_table(ours, baseline, label='ours'):
    \"\"\"Per-env summary table. R3M has no Lift number, so 'robosuite' rows have
    placeholder NaN baselines — kept in output so the env still appears in plots.\"\"\"
    rows = []
    for env in sorted(ours.get('env', {}).keys()):
        o   = ours['env'][env]['mean']
        ose = ours['env'][env]['stderr']
        if env in baseline['aggregate']:
            b   = baseline['aggregate'][env]['mean']
            bse = baseline['aggregate'][env]['stderr']
            delta = o - b
        else:
            b = float('nan'); bse = float('nan'); delta = float('nan')
        rows.append({'env': env, f'{label}_mean': o, 'r3m_mean': b, 'delta_vs_r3m': delta,
                     f'{label}_stderr': ose, 'r3m_stderr': bse})
    return rows

def plot_three_way_bars(b3_rows, b2_summary_path, baseline, out_path):
    \"\"\"Side-by-side: R3M | Baseline 2 (if results.json present) | Baseline 3.\"\"\"
    envs = [r['env'] for r in b3_rows]
    b3   = [r['ours_mean'] for r in b3_rows]
    r3m  = [r['r3m_mean']  for r in b3_rows]
    # Optionally read baseline 2's summary from disk
    b2 = [None] * len(envs)
    if b2_summary_path and Path(b2_summary_path).exists():
        try:
            b2_data = json.loads(Path(b2_summary_path).read_text())
            for i, e in enumerate(envs):
                if e in b2_data.get('summary', {}).get('ours', {}).get('env', {}):
                    b2[i] = b2_data['summary']['ours']['env'][e]['mean']
        except Exception as e:
            log(f"could not read baseline 2 summary: {e}")

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(envs)); w = 0.27
    ax.bar(x - w, r3m, w, label='R3M (paper)',         color='#777')
    if any(v is not None for v in b2):
        ax.bar(x,        [v if v is not None else 0 for v in b2],
                w, label='Baseline 2 (DINOv3+LoRA)', color='#3a8')
    ax.bar(x + w, b3, w, label='Baseline 3 (DINOv3+SAM3)', color='#e76')
    ax.set_xticks(x); ax.set_xticklabels(envs)
    ax.set_ylabel('Success rate (%)'); ax.legend(); ax.grid(axis='y', alpha=0.3)
    ax.set_title('Per-env: R3M vs Baseline 2 vs Baseline 3')
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)

def plot_per_task(ours, baseline, out_path):
    envs = sorted({e for (e, _) in ours.get('task', {}).keys()})
    if not envs:
        fig, ax = plt.subplots(); ax.set_title('no task data'); fig.savefig(out_path); plt.close(fig); return
    fig, axes = plt.subplots(1, max(1, len(envs)), figsize=(5 * max(1, len(envs)), 4), squeeze=False)
    for ax, env in zip(axes[0], envs):
        tasks = sorted([t for (e, t) in ours['task'].keys() if e == env])
        ours_vals = [ours['task'][(env, t)]['mean'] for t in tasks]
        r3m_vals  = [baseline.get('per_task_approx', {}).get(env, {}).get(t, np.nan) for t in tasks]
        x = np.arange(len(tasks)); w = 0.4
        ax.bar(x - w/2, ours_vals, w, label='b3', color='#e76')
        ax.bar(x + w/2, r3m_vals,  w, label='R3M (approx)', color='#777')
        ax.set_xticks(x); ax.set_xticklabels(tasks, rotation=30, ha='right', fontsize=8)
        ax.set_title(env); ax.set_ylabel('Success %'); ax.grid(axis='y', alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
""")


# ============================================================
# Cell 20 — auto markdown summary
# ============================================================
code("""# ===== CELL 20: auto-generated markdown summary =====
def make_summary_md(ours, delta_rows, baseline):
    lines = []
    lines.append("# Baseline 3: DINOv2 + SAM 2 — Results")
    lines.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')}_\\n")
    lines.append("## Per-environment delta (b3 − R3M)\\n")
    lines.append("| Environment | b3 mean (%) | R3M mean (%) | Δ (pp) | b3 s.e. | R3M s.e. |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in delta_rows:
        lines.append(f"| {r['env']} | {r['ours_mean']:.1f} | {r['r3m_mean']:.1f} | "
                     f"{r['delta_vs_r3m']:+.1f} | {r['ours_stderr']:.1f} | {r['r3m_stderr']:.1f} |")
    if delta_rows:
        all_o = np.mean([r['ours_mean'] for r in delta_rows])
        all_r = np.mean([r['r3m_mean']  for r in delta_rows])
        lines.append(f"| **avg** | **{all_o:.1f}** | **{all_r:.1f}** | **{all_o-all_r:+.1f}** | | |")
    lines.append("\\n## Per-task (b3)\\n")
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
    lines.append("\\n## Reproduction notes\\n")
    lines.append(f"- Hardware: 1× A100 80GB; bf16.")
    lines.append(f"- Backbone: DINOv2 {CONFIG['dinov2_variant']} (frozen).")
    lines.append(f"- Concept masking: SAM 3 with task-specific text prompts; "
                 f"mask refresh = {CONFIG['sam_refresh_steps']} steps (0 = once per episode).")
    lines.append(f"- Policy: 3-layer MLP {CONFIG['bc_hidden']} + LayerNorm.")
    lines.append(f"- BC: lr {CONFIG['bc_lr']}, batch {CONFIG['bc_batch']}, "
                 f"{CONFIG['bc_steps']:,} steps, eval every {CONFIG['bc_eval_every']:,}, "
                 f"{CONFIG['bc_eval_episodes']} eval episodes, "
                 f"plateau patience {CONFIG['bc_early_stop_patience']}.")
    lines.append(f"- Cut level: {CONFIG['cut_level']}.")
    lines.append("- Plots: `side_by_side_3way.png`, `per_task.png`. JSON: `results/results.json`.")
    return "\\n".join(lines)
""")


# ============================================================
# SMOKE TESTS — same pattern as baseline 2; stubs DINOv2 + SAM 2
# ============================================================
md("""---

## Smoke tests

Runs at micro-scale (1 env × 1 task × 1 view × 1 seed × tiny demo set) with
**stubbed DINOv2 and SAM 2** — verifies the pipeline plumbing without
requiring gated foundation-model weights. Expected runtime ~30 sec on CPU.

If this passes, the production cells are safe to run on the A100 box (where
real DINOv2 + SAM 2 will be loaded with their gated weights).

Set `RUN_SMOKE_TESTS = False` to skip on subsequent runs.
""")

code("""# ===== SMOKE 1: synthetic data + SMOKE_CONFIG =====
RUN_SMOKE_TESTS = True

if RUN_SMOKE_TESTS:
    import shutil, types
    from PIL import Image
    import numpy as np

    SMOKE_DIR = Path('./_smoke_b3')
    shutil.rmtree(SMOKE_DIR, ignore_errors=True)
    (SMOKE_DIR / 'cache').mkdir(parents=True)
    (SMOKE_DIR / 'results').mkdir(parents=True)

    SMOKE_ENV, SMOKE_TASK, SMOKE_VIEW = 'robosuite', 'Lift', 'view1'
    demo_dir = SMOKE_DIR / 'demos' / SMOKE_ENV / SMOKE_TASK / SMOKE_VIEW
    demo_dir.mkdir(parents=True, exist_ok=True)
    for i in range(6):
        T_steps = 7
        pickle.dump({
            'images':  (np.random.rand(T_steps, 64, 64, 3) * 255).astype(np.uint8),
            'actions': np.random.randn(T_steps, 4).astype(np.float32),
            'proprio': np.random.randn(T_steps, 8).astype(np.float32),
            'success': True,
        }, open(demo_dir / f"demo_{i}.pkl", 'wb'))

    SMOKE_CONFIG = copy.deepcopy(CONFIG)
    SMOKE_CONFIG.update({
        'demo_root':    str(SMOKE_DIR / 'demos'),
        'cache_dir':    str(SMOKE_DIR / 'cache'),
        'results_dir':  str(SMOKE_DIR / 'results'),
        'image_size':   64,                   # 64/16 = 4 patches per side, fast
        'feature_res':  4,
        'feat_dim':     32,                   # tiny for smoke speed
        'bc_steps':     50,
        'bc_eval_every':25,
        'bc_batch':     8,
        'bc_eval_episodes': 2,
        'bc_early_stop_patience': 10,
        'envs': {SMOKE_ENV: {'tasks': [SMOKE_TASK], 'views': [SMOKE_VIEW],
                              'demo_sizes': [5], 'horizon': 5}},
        'seeds': [0],
        'compile_model': False,
        'num_workers': 0,
    })
    log(f"smoke setup OK: {SMOKE_DIR.resolve()}")
else:
    log("RUN_SMOKE_TESTS=False — skipping")
""")

code("""# ===== SMOKE 2: end-to-end mini pipeline (stubbed DINOv2 + SAM 2) =====
if RUN_SMOKE_TESTS:
    # ---- stub DINOv3 ----
    class _StubDinov2Backbone(nn.Module):
        \"\"\"Mimics DINOv3.get_intermediate_layers(reshape=True) returning (B, D, h, w).
        kernel/stride = image_size / feature_res (= patch size for the real ViT).\"\"\"
        def __init__(self, d, image_size, feature_res):
            super().__init__()
            patch = image_size // feature_res
            self.proj = nn.Conv2d(3, d, kernel_size=patch, stride=patch)
            self._bulk = nn.Parameter(torch.zeros(1024, 1024))   # ~1M frozen params
        def get_intermediate_layers(self, x, *, n=1, reshape=True,
                                     return_class_token=False, norm=True):
            return [self.proj(x)]   # (B, D, h, w)

    _real_load_dinov2 = load_dinov2_backbone
    globals()['load_dinov2_backbone'] = lambda variant: _StubDinov2Backbone(
        SMOKE_CONFIG['feat_dim'], SMOKE_CONFIG['image_size'], SMOKE_CONFIG['feature_res']
    )

    # ---- stub SAM 2 ----
    class _StubSam2:
        def __init__(self, cfg): self.cfg = cfg
        def mask(self, img_uint8, feature_res):
            # Deterministic fake mask: a centered Gaussian blob, thresholded
            r = feature_res
            cy, cx = r // 2, r // 2
            yy, xx = torch.meshgrid(torch.arange(r), torch.arange(r), indexing='ij')
            blob = torch.exp(-((yy - cy)**2 + (xx - cx)**2) / max(1, r//2))
            return (blob > 0.5).float()

    _real_sam2_cls = Sam2Masker
    globals()['Sam2Masker'] = _StubSam2

    # ---- 1. dense encoder smoke ----
    dinov2 = DinoV2DenseEncoder(SMOKE_CONFIG).to(DEVICE)
    test_img = (np.random.rand(SMOKE_CONFIG['image_size'], SMOKE_CONFIG['image_size'], 3)*255).astype(np.uint8)
    fmap = encode_image_to_feature_map(dinov2, test_img, SMOKE_CONFIG['image_size'])
    assert fmap.shape == (SMOKE_CONFIG['feat_dim'], SMOKE_CONFIG['feature_res'], SMOKE_CONFIG['feature_res']), fmap.shape
    log(f"smoke 1/7: DinoV2DenseEncoder OK ({tuple(fmap.shape)})")

    # ---- 2. SAM 2 mask smoke ----
    sam2 = _StubSam2(SMOKE_CONFIG)
    m = sam2.mask(test_img, SMOKE_CONFIG['feature_res'])
    assert m.shape == (SMOKE_CONFIG['feature_res'], SMOKE_CONFIG['feature_res']), m.shape
    assert (m.sum() > 0), "stub mask is all zeros"
    log(f"smoke 2/7: Sam2Masker OK (mask shape {tuple(m.shape)}, fg-pixels {int(m.sum())})")

    # ---- 3. masked pooling ----
    z = masked_feature_pool(fmap.to(DEVICE), m.to(DEVICE))
    assert z.shape == (SMOKE_CONFIG['feat_dim'],), z.shape
    log(f"smoke 3/7: masked_feature_pool OK (z {tuple(z.shape)})")

    # ---- 4. demo feature extraction ----
    feats = encode_episode_features(dinov2, sam2,
                                     (np.random.rand(7, SMOKE_CONFIG['image_size'],
                                                     SMOKE_CONFIG['image_size'], 3)*255).astype(np.uint8),
                                     SMOKE_CONFIG)
    assert feats.shape == (7, SMOKE_CONFIG['feat_dim']), feats.shape
    log(f"smoke 4/7: encode_episode_features OK ({tuple(feats.shape)})")

    # ---- 5. extract_features_for_env over synthetic demos ----
    extract_features_for_env(SMOKE_CONFIG, dinov2, sam2, SMOKE_ENV,
                             SMOKE_CONFIG['envs'][SMOKE_ENV])
    cf = Path(SMOKE_CONFIG['cache_dir']) / 'features' / SMOKE_ENV / f"{SMOKE_TASK}__{SMOKE_VIEW}.pt"
    assert cf.exists(), cf
    log(f"smoke 5/7: feature extraction OK -> {cf.name}")

    # ---- 6. BC dataset + policy ----
    bc_ds = FeatureBCDataset(cf, n_demos=5)
    in_dim = bc_ds.feats.shape[1] + bc_ds.proprio.shape[1]
    pol = BCPolicyB3(in_dim, bc_ds.actions.shape[1], hidden=(32, 32, 32),
                      use_layernorm=SMOKE_CONFIG['bc_use_layernorm'])
    xx, aa = next(iter(DataLoader(bc_ds, batch_size=8)))
    pred = pol(xx); assert pred.shape == aa.shape
    log(f"smoke 6/7: BCPolicyB3 OK in_dim={in_dim} act_dim={aa.shape[1]}")

    # ---- 7. BC train with stubbed sim env ----
    _real_make_env = make_env
    _real_rollout  = rollout
    def _stub_make_env(env_name, task, view, image_size=224):
        return types.SimpleNamespace(close=lambda: None)
    def _stub_rollout(env, env_name, dinov2, sam2, policy, horizon,
                       sam_refresh_steps=0, image_size=224):
        return random.random() > 0.5
    globals()['make_env'] = _stub_make_env
    globals()['rollout']  = _stub_rollout
    try:
        SMOKE_CONFIG['bc_hidden'] = (32, 32, 32)
        r = train_bc_one(SMOKE_CONFIG, SMOKE_ENV, SMOKE_TASK, SMOKE_VIEW, 5, 0, dinov2, sam2)
        assert r['best_success'] is not None
        assert len(r['history']) > 0
    finally:
        globals()['make_env'] = _real_make_env
        globals()['rollout']  = _real_rollout
    log(f"smoke 7/7: BC train (stubbed sim) OK best={r['best_success']:.1f}%")

    # ---- 8. aggregation + plots + summary ----
    fake_sweep = [
        {'env': 'robosuite', 'task': 'Lift', 'view': 'view1',
         'n_demos': 10, 'seed': 0, 'best_success': 80.0, 'history': []},
        {'env': 'robosuite', 'task': 'Lift', 'view': 'view1',
         'n_demos': 50, 'seed': 0, 'best_success': 95.0, 'history': []},
    ]
    ours = summarize(fake_sweep)
    rows = delta_table(ours, R3M_BASELINE, label='ours')
    assert len(rows) >= 1
    plot_three_way_bars(rows, None, R3M_BASELINE,
                          Path(SMOKE_CONFIG['results_dir']) / 'sbs.png')
    plot_per_task(ours, R3M_BASELINE, Path(SMOKE_CONFIG['results_dir']) / 'pt.png')
    md_text = make_summary_md(ours, rows, R3M_BASELINE)
    assert "Per-environment delta" in md_text
    log("smoke aggregation + plots + summary OK")

    print()
    print("=" * 60)
    print("=== ALL B3 SMOKE TESTS PASSED ===")
    print("Pipeline plumbing works; safe to run production cells with real DINOv2 + SAM 2.")
    print("=" * 60)
""")

code("""# ===== SMOKE 3: cleanup =====
if RUN_SMOKE_TESTS:
    import shutil
    shutil.rmtree('./_smoke_b3', ignore_errors=True)
    if '_real_load_dinov2' in globals(): globals()['load_dinov2_backbone'] = _real_load_dinov2
    if '_real_sam2_cls'    in globals(): globals()['Sam2Masker']    = _real_sam2_cls
    log("smoke artifacts removed; production caches untouched; DINOv2/SAM 2 stubs restored")
""")


# ============================================================
# PRODUCTION
# ============================================================
md("""---

## Production run

| Cell | What it does | Skip condition |
|---|---|---|
| PROD 1 | Pre-extract DINOv3 features + SAM 3 masks per (task, view) | per-(task, view) cache hit |
| PROD 2 | BC sweep | always runs |
| PROD 3 | Aggregate + plots | always runs |
| PROD 4 | Markdown summary | always runs |

**Loading order:** DINOv2 + SAM 2 are loaded once at the start of PROD 1 and
reused for the BC sweep. Both have gated weights — the cells will fail loudly
if access hasn't been granted.
""")

code("""# ===== PROD 1: pre-extract features + masks =====
DINOV3_MODEL, SAM3_MODEL = run_feature_extraction(CONFIG)
""")

code("""# ===== PROD 2: BC sweep =====
SWEEP = run_bc_sweep(CONFIG, DINOV3_MODEL, SAM3_MODEL)
""")

code("""# ===== PROD 3: aggregate, delta, plots =====
if SWEEP:
    OURS_SUMMARY = summarize(SWEEP)
    DELTA_ROWS = delta_table(OURS_SUMMARY, R3M_BASELINE, label='ours')
    plots = Path(CONFIG['results_dir'])
    # Optionally pull baseline 2's results.json for the 3-way plot
    b2_results = Path(CONFIG['cache_dir']).parent / 'results' / 'results.json'   # baseline 2's default
    plot_three_way_bars(DELTA_ROWS, str(b2_results), R3M_BASELINE,
                         plots / 'side_by_side_3way.png')
    plot_per_task(OURS_SUMMARY, R3M_BASELINE, plots / 'per_task.png')
    results.log('summary', 'ours', OURS_SUMMARY)
    results.log('summary', 'delta', DELTA_ROWS)
    log(f"plots written -> {plots}/*.png")
else:
    OURS_SUMMARY, DELTA_ROWS = {}, []
""")

code("""# ===== PROD 4: markdown summary =====
SUMMARY_MD = make_summary_md(OURS_SUMMARY, DELTA_ROWS, R3M_BASELINE)
out_md = Path(CONFIG['results_dir']) / 'summary.md'
out_md.write_text(SUMMARY_MD)
log(f"summary -> {out_md}")
print(); print(SUMMARY_MD)
""")


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
NB_PATH.write_text(json.dumps(nb, indent=1))
print(f"wrote {NB_PATH} ({len(cells)} cells)")
