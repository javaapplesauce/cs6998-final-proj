"""Build baseline_3_dinov3_sam3.ipynb — DINOv3 + SAM 3 frozen, no Ego4D pretraining."""
import json, uuid
from pathlib import Path

NB_PATH = Path(__file__).parent / "baseline_3_dinov3_sam3.ipynb"
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
md("""# Baseline 3: DINOv3 + SAM 3 (frozen) vs. R3M

Frozen-foundation-model approach: no Ego4D pretraining, no LoRA. DINOv3 ViT-B/16
provides dense spatial features; SAM 3 provides text-prompted concept masks
that filter the feature map element-wise before global average pooling. Only
a small BC MLP head is trainable.

## Comparison framing

- **Baseline 1** — R3M (Nair et al., 2022): ResNet50 with TCN+VLA+L1/L2 on Ego4D 1M steps. Numbers hardcoded from paper Tables 1-3.
- **Baseline 2** (`baseline_2_dinov3_lora.ipynb`): DINOv3 ViT-B/16 + LoRA r=16, R3M's losses on Ego4D 150k steps.
- **Baseline 3** (this notebook): DINOv3 ViT-B/16 + SAM 3, fully frozen, **no Ego4D pretraining**, BC MLP only.

The hypothesis: **frozen foundation models (DINOv3 + SAM 3 with concept-mask
filtering) suffice for manipulation BC, without the contrastive Ego4D
pretraining that Baseline 2 inherits from R3M**. If Baseline 3 ≥ Baseline 2,
Ego4D pretraining is redundant given DINOv3's strong init + SAM's grounding.

**Hardware target:** 1× A100 80GB, ≤ 4 h wall clock.
""")


# ============================================================
# Cell 2 — bottleneck analysis
# ============================================================
md("""## Bottleneck analysis & wall-clock budget

| Stage | Configuration | Estimate |
|---|---|---|
| Demo feature pre-extraction (DINOv3 dense map) | 12 task-views × ~10k frames × ~3 ms | ~10 min |
| Demo mask pre-extraction (SAM 3, text-prompted) | 12 task-views × ~10k frames × ~50 ms | ~100 min |
| BC sweep | 5+5 tasks × 2 views × 2 sizes × 3 seeds = 120 runs | ~80-120 min |
| Aggregate + plots + summary | | ~2 min |
| **Total** | | **~3.2-3.9 h** |

**Why no pretrain stage:** both DINOv3 and SAM 3 are loaded already-trained from foundation-model weights. We only optimize a 2-3 layer MLP policy head (~200K params) on top.

**Top 3 bottlenecks**
1. **SAM 3 forward at every demo frame.** Mitigation: pre-extract once per (task, view); cache binary masks at feature-map resolution (14×14) instead of original (224×224) — 256× smaller on disk.
2. **Live SAM 3 inference at BC eval rollouts.** Real-time SAM at every rollout step is impossibly slow (50 ms × 200 step horizons × 200 evals = hours per BC run). Mitigation: SAM runs once at episode start and the mask is reused across the episode. Configurable via `CONFIG['sam_refresh_steps']`.
3. **DINOv3 ViT-L is too big for budget.** Proposal specifies ViT-L/16. We default to ViT-B/16 for budget parity with baseline 2; user can override `CONFIG['dinov3_variant'] = 'dinov3_vitl16'` at the cost of 3-4× slower forward.

**Cut order (`CONFIG['cut_level']`):** Same as baseline 2 — drop Adroit → demo sizes 3→2 → seeds 3→2 → drop view 3.
""")


# ============================================================
# Cell 3 — environment setup
# ============================================================
md("""## Environment setup

Required packages (in addition to baseline 2's core deps):

```bash
# DINOv3 + SAM 3 are gated foundation-model weights; you must request access:
#   - DINOv3:  https://ai.meta.com/dino-v3        (URL-emailed weights)
#   - SAM 3:   https://huggingface.co/facebook/sam3  (HF gated; `hf auth login` after acceptance)
git clone https://github.com/facebookresearch/sam3.git
cd sam3 && pip install -e . && cd ..
pip install transformers>=4.56.0   # used for DINOv3 via HF as a fallback if torch.hub weights URL is not set
```

The smoke section stubs both DINOv3 and SAM 3 — you can run smoke without weight access.
""")


# ============================================================
# Cell 4 — CONFIG + imports + setup
# ============================================================
code("""# ===== CELL 4: CONFIG + imports + setup =====
import os, sys, json, time, math, random, hashlib, pickle, shutil, copy
from pathlib import Path

CONFIG = {
    # ---- paths ----
    'demo_root':     '/data/r3m_demos',                  # TODO: confirm — same demo .pkl tree baseline 2 uses
    'cache_dir':     './cache_b3',
    'results_dir':   './results_b3',

    # ---- model ----
    'dinov3_variant':   'dinov3_vitb16',           # ViT-B/16 default for budget; vitl16 for proposal-spec at 3-4× cost
    'dinov3_weights':   None,                      # gated; URL or local path; required for production
    'feat_dim':         768,                       # 768 for vitb16, 1024 for vitl16
    'sam3_checkpoint':  None,                      # set to local path or leave None for `hf auth login` HF-cached default
    'image_size':       224,
    'feature_res':      14,                        # 224/16 = 14 patches per side
    'mask_threshold':   0.5,                       # binarize SAM 3 score per pixel

    # ---- BC eval (R3M protocol, Appendix B.5; preserved for direct comparability) ----
    'bc_lr':            1e-3,
    'bc_batch':         32,
    'bc_steps':         20_000,
    'bc_eval_every':    1_000,
    'bc_hidden':        (256, 256, 256),           # 3-layer MLP per proposal §3.1 (vs baseline 2's 2-layer)
    'bc_use_layernorm': True,                      # proposal-spec; baseline 2 uses BatchNorm at input
    'bc_eval_episodes': 10,
    'bc_early_stop_patience': 3,

    # ---- SAM 3 eval policy ----
    'sam_refresh_steps': 0,                        # 0 = run once per episode; >0 = re-run every N steps in eval

    # ---- envs (same scope as baseline 2 for direct comparison) ----
    'envs': {
        'franka_kitchen': {
            'tasks': ['slide', 'light', 'door', 'knob', 'micro'],
            'views': ['view1', 'view2', 'view3'],   # # TODO: confirm
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
    'cut_level': 1,    # 0=full sweep; 1=drop Adroit; 2=demo sizes 3→2; 3=seeds 3→2; 4=drop view 3

    # ---- compute ----
    'precision':     'bf16',
    'compile_model': False,                        # skip compile by default — DINOv3+SAM3 dynamic shapes hurt compile cache hit
    'num_workers':   4,
    'pin_memory':    True,
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

print(f"[setup] device={DEVICE}, dtype={DTYPE}, cut_level={CONFIG['cut_level']}, "
      f"envs={list(CONFIG['envs'].keys())}, dinov3={CONFIG['dinov3_variant']}")
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

# SAM 3 + DINOv3-via-transformers are heavyweight; check presence loudly.
REQUIRED = {'torch':'torch>=2.7', 'torchvision':'torchvision', 'PIL':'pillow',
            'numpy':'numpy', 'pandas':'pandas', 'matplotlib':'matplotlib',
            'transformers':'transformers>=4.56'}
missing = [pkg for mod, pkg in REQUIRED.items() if importlib.util.find_spec(mod) is None]
if missing:
    print(f"[warn] missing: pip install {' '.join(missing)}")
else:
    print("[ok] core packages present")
for opt in ('sam3', 'mj_envs', 'metaworld'):
    if importlib.util.find_spec(opt) is None:
        print(f"[warn] {opt} not importable — required for production; smoke section stubs it")
    else:
        print(f"[ok] {opt} importable")

if IN_COLAB or DRIVE_ROOT.exists():
    DATA_ROOT = DRIVE_ROOT / 'cs6998-final-proj-data'
    RUNS_ROOT = DRIVE_ROOT / 'cs6998-final-proj-runs'
    CONFIG.update({
        'demo_root':    str(DATA_ROOT / 'r3m_demos'),       # TODO: confirm
        'cache_dir':    str(RUNS_ROOT / 'cache_b3'),
        'results_dir':  str(RUNS_ROOT / 'results_b3'),
    })
    Path(CONFIG['cache_dir']).mkdir(parents=True, exist_ok=True)
    Path(CONFIG['results_dir']).mkdir(parents=True, exist_ok=True)
    print(f"\\n[ok] CONFIG paths patched to Drive (cache_dir={CONFIG['cache_dir']})")
else:
    print(f"\\n[info] DRIVE_ROOT={DRIVE_ROOT} not accessible — leaving Cell 4 defaults intact.")
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
            self.data = {'baseline_id': 'b3_dinov3_sam3', 'config': CONFIG,
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
code("""# ===== CELL 9: task → SAM 3 text prompt mapping =====
# # TODO: confirm — adapt these to what your sim's task-relevant object actually looks like.
# SAM 3 takes a short concept phrase. Less is more — \"the red mug\" beats \"the small red ceramic mug to the left\".
TASK_TEXT_PROMPTS = {
    'franka_kitchen': {
        'slide':  'the sliding cabinet door',
        'light':  'the kitchen light switch',
        'door':   'the cabinet door handle',
        'knob':   'the stove knob',
        'micro':  'the microwave door',
    },
    'metaworld': {
        'assembly-v2':              'the green ring',
        'bin-picking-v2':           'the red block',
        'button-press-topdown-v2':  'the red button',
        'drawer-open-v2':           'the wooden drawer handle',
        'hammer-v2':                'the red hammer',
    },
    'adroit': {
        'pen-v0':       'the blue pen',
        'relocate-v0':  'the red ball',
    },
}
log(f"loaded text prompts for {sum(len(v) for v in TASK_TEXT_PROMPTS.values())} tasks")
""")


# ============================================================
# Cell 10 — DINOv3 dense feature extractor (frozen)
# ============================================================
code("""# ===== CELL 10: DINOv3 dense feature extractor (frozen) =====
# Baseline 3 uses DINOv3's *patch tokens* (h × w × d), not just the CLS token —
# spatial structure is preserved so SAM 3's mask can do something meaningful at
# the feature-map resolution. We use get_intermediate_layers(reshape=True) to
# get a (B, D, h, w) tensor.

def load_dinov3_backbone(variant: str, weights):
    if weights is None:
        raise RuntimeError(
            "CONFIG['dinov3_weights'] is None. DINOv3 weights are gated — request "
            "access at https://ai.meta.com/dino-v3 and set the URL/local-path here."
        )
    return torch.hub.load('facebookresearch/dinov3', variant,
                           source='github', weights=weights)

class DinoV3DenseEncoder(nn.Module):
    \"\"\"Frozen DINOv3. Returns (B, D, h, w) feature map.\"\"\"
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = load_dinov3_backbone(cfg['dinov3_variant'], cfg.get('dinov3_weights'))
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    @torch.no_grad()
    def forward(self, x):
        # x: (B, 3, H, W) normalized — H = W = image_size
        # get_intermediate_layers returns last block; reshape=True -> (B, D, h, w)
        out = self.backbone.get_intermediate_layers(
            x, n=1, reshape=True, return_class_token=False, norm=True
        )
        return out[0]   # (B, D, h, w)

@torch.no_grad()
def encode_image_to_feature_map(model: 'DinoV3DenseEncoder', img_uint8, image_size=224):
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
        feat = model(x)         # (1, D, h, w)
    return feat[0].float().cpu()
""")


# ============================================================
# Cell 11 — SAM 3 concept mask extractor (frozen)
# ============================================================
code("""# ===== CELL 11: SAM 3 concept mask extractor (frozen, text-prompted) =====
# We wrap SAM 3's API behind a single function that takes (image, text_prompt) and
# returns a binary mask at feature-map resolution (h × w), so it's directly
# multipliable with DINOv3's dense feature.

class Sam3ConceptMasker:
    def __init__(self, cfg):
        self.cfg = cfg
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            self.model = build_sam3_image_model(checkpoint=cfg.get('sam3_checkpoint'))
            self.processor = Sam3Processor(self.model)
            self._available = True
        except ImportError:
            raise ImportError(
                "sam3 not installed. See setup cell — clone facebookresearch/sam3 and pip install -e ."
            )
        except Exception as e:
            raise RuntimeError(
                f"SAM 3 weights load failed ({e}). Run `hf auth login` after requesting "
                f"access at https://huggingface.co/facebook/sam3."
            )
        self._available = True

    @torch.no_grad()
    def mask(self, img_uint8, text_prompt: str, feature_res: int):
        \"\"\"Returns (feature_res, feature_res) float32 mask in [0, 1] (post-binarization).
        Multiple instance masks are unioned (max).\"\"\"
        from PIL import Image as _PIL
        import numpy as np
        pil = _PIL.fromarray(img_uint8)
        state = self.processor.set_image(pil)
        out = self.processor.set_text_prompt(state=state, prompt=text_prompt)
        masks = out['masks']   # numpy or tensor; shape (N_inst, H, W) bool
        if hasattr(masks, 'cpu'): masks = masks.cpu().numpy()
        if masks.ndim == 2: masks = masks[None]
        if masks.shape[0] == 0:
            return torch.zeros(feature_res, feature_res, dtype=torch.float32)
        # Union of all instance masks
        union = (masks.max(axis=0) > 0).astype('float32')   # (H, W)
        # Downsample to feature-map resolution via average pool, then threshold
        m = torch.from_numpy(union).unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)
        m = F.adaptive_avg_pool2d(m, (feature_res, feature_res))[0, 0]
        return (m > self.cfg['mask_threshold']).float()
""")


# ============================================================
# Cell 12 — combined feature extractor (DINOv3 + mask + GAP)
# ============================================================
code("""# ===== CELL 12: combined visual representation: DINOv3 dense ⊙ SAM 3 mask -> GAP =====
def masked_feature_pool(feat_map, mask):
    \"\"\"feat_map: (D, h, w); mask: (h, w) in [0,1] -> z_vis: (D,)
    Empty-mask fallback: average pool unmasked features (so the run doesn't NaN out
    if SAM finds zero pixels).\"\"\"
    if mask.sum() < 1.0:
        return feat_map.mean(dim=(1, 2))
    masked = feat_map * mask.unsqueeze(0)
    return masked.sum(dim=(1, 2)) / mask.sum().clamp_min(1.0)

@torch.no_grad()
def encode_episode_features(dinov3, sam3, images_uint8, text_prompt, cfg):
    \"\"\"Returns per-frame z_vis tensor: (T, D).\"\"\"
    feats = []
    for img in images_uint8:
        feat_map = encode_image_to_feature_map(dinov3, img, cfg['image_size'])  # (D, h, w)
        mask = sam3.mask(img, text_prompt, cfg['feature_res'])                  # (h, w)
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
code("""# ===== CELL 14: simulator wrappers (same env API as baseline 2) =====
def make_env(env_name, task, view, image_size=224):
    if env_name == 'metaworld':
        try:
            from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as REG
            cls = REG[f"{task}-goal-observable"]
        except (ImportError, KeyError):
            from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS as REG
            cls = REG[task]
        env = cls()
        env._partially_observable = False
        view_to_camera = {'view1': 'corner', 'view2': 'corner2', 'view3': 'topview'}   # # TODO: confirm
        env._camera_name = view_to_camera.get(view, 'corner')
        return env
    elif env_name == 'franka_kitchen':
        import gym, mj_envs   # noqa: F401
        task_to_id = {'slide':'kitchen_sdoor_open-v3','light':'kitchen_light_on-v3',
                      'door':'kitchen_ldoor_open-v3','knob':'kitchen_knob1_on-v3',
                      'micro':'kitchen_micro_open-v3'}
        return gym.make(task_to_id[task])
    elif env_name == 'adroit':
        import gym, mj_envs   # noqa: F401
        return gym.make({'pen-v0':'pen-v0','relocate-v0':'relocate-v0'}[task])
    raise ValueError(f"unknown env {env_name}")

def render_rgb(env, image_size=224):
    for kwargs in ({'mode':'rgb_array','width':image_size,'height':image_size},
                   {'mode':'rgb_array'},
                   {'width':image_size,'height':image_size},
                   {}):
        try: return env.render(**kwargs)
        except TypeError: continue
    raise RuntimeError("env.render produced no rgb image")

def get_proprio(env, env_name):
    # # TODO: confirm proprio composition matches baseline 2 / R3M
    if env_name == 'metaworld':       return env._get_obs()[:8]
    if env_name == 'franka_kitchen':  return env.unwrapped._get_obs()[:9]
    if env_name == 'adroit':          return env.unwrapped._get_obs()[:30]
    return env._get_obs()
""")


# ============================================================
# Cell 15 — rollout (live DINOv3 + SAM at episode start)
# ============================================================
code("""# ===== CELL 15: rollout with frozen DINOv3 + SAM 3 (mask cached per episode) =====
@torch.no_grad()
def rollout(env, env_name, dinov3, sam3, policy, horizon, text_prompt,
            sam_refresh_steps=0, image_size=224):
    \"\"\"
    sam_refresh_steps=0  -> SAM 3 runs once at frame 0; mask cached for the episode.
    sam_refresh_steps=K  -> SAM 3 re-runs every K steps (linearly more expensive).
    \"\"\"
    from PIL import Image
    import numpy as np
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    success = False; cached_mask = None

    for t in range(horizon):
        rgb = render_rgb(env, image_size)
        if rgb.shape[0] != image_size:
            rgb = np.asarray(Image.fromarray(rgb).resize((image_size, image_size), Image.Resampling.BICUBIC))

        # SAM 3: refresh mask at episode start, optionally every K steps
        if cached_mask is None or (sam_refresh_steps > 0 and t % sam_refresh_steps == 0):
            cached_mask = sam3.mask(rgb, text_prompt, CONFIG['feature_res']).to(DEVICE)

        # DINOv3 every step
        feat_map = encode_image_to_feature_map(dinov3, rgb, image_size).to(DEVICE)  # (D, h, w)
        z_vis    = masked_feature_pool(feat_map, cached_mask)                       # (D,)

        p = torch.as_tensor(get_proprio(env, env_name), device=DEVICE, dtype=torch.float32)
        a = policy(torch.cat([z_vis, p], dim=-1).unsqueeze(0)).squeeze(0).cpu().numpy()
        step_out = env.step(a)
        if len(step_out) == 5:
            obs, _, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        else:
            obs, _, done, info = step_out
        if isinstance(info, dict) and (info.get('success') or info.get('goal_achieved')):
            success = True
        if done: break
    return success
""")


# ============================================================
# Cell 16 — BC train + plateau early-stop
# ============================================================
code("""# ===== CELL 16: BC train + plateau early-stop =====
def train_bc_one(cfg, env_name, task, view, n_demos, seed, dinov3, sam3):
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
    text_prompt = TASK_TEXT_PROMPTS[env_name][task]
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
            successes = sum(rollout(env, env_name, dinov3, sam3, policy, horizon, text_prompt,
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
def run_bc_sweep(cfg, dinov3, sam3):
    sweep = []; sweep_t0 = time.time()
    for env_name, env_cfg in cfg['envs'].items():
        for task in env_cfg['tasks']:
            for view in env_cfg['views']:
                for n_demos in env_cfg['demo_sizes']:
                    for seed in cfg['seeds']:
                        key = f"{env_name}/{task}/{view}/n{n_demos}/s{seed}"
                        try:
                            r = train_bc_one(cfg, env_name, task, view, n_demos, seed, dinov3, sam3)
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
def extract_features_for_env(cfg, dinov3, sam3, env_name, env_cfg):
    out_dir = Path(cfg['cache_dir']) / 'features' / env_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for task in env_cfg['tasks']:
        text_prompt = TASK_TEXT_PROMPTS[env_name][task]
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
                feats = encode_episode_features(dinov3, sam3, d['images'], text_prompt, cfg)
                episodes.append({
                    'features': feats,
                    'actions':  torch.as_tensor(d['actions'], dtype=torch.float32),
                    'proprio':  torch.as_tensor(d['proprio'], dtype=torch.float32),
                    'success':  bool(d.get('success', True)),
                })
            torch.save({'env': env_name, 'task': task, 'view': view, 'episodes': episodes},
                        cache_file)
            log(f"saved {cache_file} ({len(episodes)} demos, prompt={text_prompt!r})")

def run_feature_extraction(cfg):
    log("loading DINOv3 + SAM 3 (gated weights)...")
    dinov3 = DinoV3DenseEncoder(cfg).to(DEVICE)
    sam3   = Sam3ConceptMasker(cfg)
    for env_name, env_cfg in cfg['envs'].items():
        extract_features_for_env(cfg, dinov3, sam3, env_name, env_cfg)
    return dinov3, sam3
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
    rows = []
    for env in ['franka_kitchen', 'metaworld', 'adroit']:
        if env not in ours['env']: continue
        o = ours['env'][env]['mean']; b = baseline['aggregate'][env]['mean']
        rows.append({'env': env, f'{label}_mean': o, 'r3m_mean': b, 'delta_vs_r3m': o - b,
                     f'{label}_stderr': ours['env'][env]['stderr'],
                     'r3m_stderr':  baseline['aggregate'][env]['stderr']})
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
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, env in zip(axes, ['franka_kitchen', 'metaworld', 'adroit']):
        tasks = sorted([t for (e, t) in ours['task'].keys() if e == env])
        if not tasks: ax.set_title(f"{env} (no data)"); continue
        ours_vals = [ours['task'][(env, t)]['mean'] for t in tasks]
        r3m_vals  = [baseline['per_task_approx'][env].get(t, np.nan) for t in tasks]
        x = np.arange(len(tasks)); w = 0.4
        ax.bar(x - w/2, ours_vals, w, label='b3', color='#e76')
        ax.bar(x + w/2, r3m_vals,  w, label='R3M (Fig 9 approx)', color='#777')
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
    lines.append("# Baseline 3: DINOv3 + SAM 3 — Results")
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
    lines.append(f"- Backbone: DINOv3 {CONFIG['dinov3_variant']} (frozen).")
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
# SMOKE TESTS — same pattern as baseline 2; stubs DINOv3 + SAM 3
# ============================================================
md("""---

## Smoke tests

Runs at micro-scale (1 env × 1 task × 1 view × 1 seed × tiny demo set) with
**stubbed DINOv3 and SAM 3** — verifies the pipeline plumbing without
requiring gated foundation-model weights. Expected runtime ~30 sec on CPU.

If this passes, the production cells are safe to run on the A100 box (where
real DINOv3 + SAM 3 will be loaded with their gated weights).

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

    SMOKE_ENV, SMOKE_TASK, SMOKE_VIEW = 'metaworld', 'assembly-v2', 'view1'
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

code("""# ===== SMOKE 2: end-to-end mini pipeline (stubbed DINOv3 + SAM 3) =====
if RUN_SMOKE_TESTS:
    # ---- stub DINOv3 ----
    class _StubDinov3Backbone(nn.Module):
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

    _real_load_dinov3 = load_dinov3_backbone
    globals()['load_dinov3_backbone'] = lambda variant, weights: _StubDinov3Backbone(
        SMOKE_CONFIG['feat_dim'], SMOKE_CONFIG['image_size'], SMOKE_CONFIG['feature_res']
    )

    # ---- stub SAM 3 ----
    class _StubSam3:
        def __init__(self, cfg): self.cfg = cfg
        def mask(self, img_uint8, text_prompt, feature_res):
            # Deterministic fake mask: a centered Gaussian-shaped blob, thresholded
            r = feature_res
            cy, cx = r // 2, r // 2
            yy, xx = torch.meshgrid(torch.arange(r), torch.arange(r), indexing='ij')
            blob = torch.exp(-((yy - cy)**2 + (xx - cx)**2) / max(1, r//2))
            return (blob > 0.5).float()

    _real_sam3_cls = Sam3ConceptMasker
    globals()['Sam3ConceptMasker'] = _StubSam3

    # ---- 1. dense encoder smoke ----
    dinov3 = DinoV3DenseEncoder(SMOKE_CONFIG).to(DEVICE)
    test_img = (np.random.rand(SMOKE_CONFIG['image_size'], SMOKE_CONFIG['image_size'], 3)*255).astype(np.uint8)
    fmap = encode_image_to_feature_map(dinov3, test_img, SMOKE_CONFIG['image_size'])
    assert fmap.shape == (SMOKE_CONFIG['feat_dim'], SMOKE_CONFIG['feature_res'], SMOKE_CONFIG['feature_res']), fmap.shape
    log(f"smoke 1/7: DinoV3DenseEncoder OK ({tuple(fmap.shape)})")

    # ---- 2. SAM 3 mask smoke ----
    sam3 = _StubSam3(SMOKE_CONFIG)
    m = sam3.mask(test_img, 'a green ring', SMOKE_CONFIG['feature_res'])
    assert m.shape == (SMOKE_CONFIG['feature_res'], SMOKE_CONFIG['feature_res']), m.shape
    assert (m.sum() > 0), "stub mask is all zeros"
    log(f"smoke 2/7: Sam3ConceptMasker OK (mask shape {tuple(m.shape)}, fg-pixels {int(m.sum())})")

    # ---- 3. masked pooling ----
    z = masked_feature_pool(fmap.to(DEVICE), m.to(DEVICE))
    assert z.shape == (SMOKE_CONFIG['feat_dim'],), z.shape
    log(f"smoke 3/7: masked_feature_pool OK (z {tuple(z.shape)})")

    # ---- 4. demo feature extraction ----
    SMOKE_TEXT_PROMPT = 'a green ring'
    feats = encode_episode_features(dinov3, sam3,
                                     (np.random.rand(7, SMOKE_CONFIG['image_size'],
                                                     SMOKE_CONFIG['image_size'], 3)*255).astype(np.uint8),
                                     SMOKE_TEXT_PROMPT, SMOKE_CONFIG)
    assert feats.shape == (7, SMOKE_CONFIG['feat_dim']), feats.shape
    log(f"smoke 4/7: encode_episode_features OK ({tuple(feats.shape)})")

    # ---- 5. extract_features_for_env over synthetic demos ----
    # Need TASK_TEXT_PROMPTS to have an entry for the smoke task
    _orig_prompts = TASK_TEXT_PROMPTS
    globals()['TASK_TEXT_PROMPTS'] = {SMOKE_ENV: {SMOKE_TASK: SMOKE_TEXT_PROMPT}}
    extract_features_for_env(SMOKE_CONFIG, dinov3, sam3, SMOKE_ENV,
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
    def _stub_rollout(env, env_name, dinov3, sam3, policy, horizon, text_prompt,
                       sam_refresh_steps=0, image_size=224):
        return random.random() > 0.5
    globals()['make_env'] = _stub_make_env
    globals()['rollout']  = _stub_rollout
    try:
        # Use a tiny BC hidden for smoke
        SMOKE_CONFIG['bc_hidden'] = (32, 32, 32)
        r = train_bc_one(SMOKE_CONFIG, SMOKE_ENV, SMOKE_TASK, SMOKE_VIEW, 5, 0, dinov3, sam3)
        assert r['best_success'] is not None
        assert len(r['history']) > 0
    finally:
        globals()['make_env']            = _real_make_env
        globals()['rollout']             = _real_rollout
        globals()['TASK_TEXT_PROMPTS']   = _orig_prompts
    log(f"smoke 7/7: BC train (stubbed sim) OK best={r['best_success']:.1f}%")

    # ---- 8. aggregation + plots + summary ----
    fake_sweep = [
        {'env': 'metaworld',      'task': 'assembly-v2', 'view': 'view1',
         'n_demos': 5, 'seed': 0, 'best_success': 65.0, 'history': []},
        {'env': 'franka_kitchen', 'task': 'slide',       'view': 'view1',
         'n_demos': 5, 'seed': 0, 'best_success': 50.0, 'history': []},
    ]
    ours = summarize(fake_sweep)
    rows = delta_table(ours, R3M_BASELINE, label='ours')
    assert len(rows) == 2
    plot_three_way_bars(rows, None, R3M_BASELINE,
                          Path(SMOKE_CONFIG['results_dir']) / 'sbs.png')
    plot_per_task(ours, R3M_BASELINE, Path(SMOKE_CONFIG['results_dir']) / 'pt.png')
    md_text = make_summary_md(ours, rows, R3M_BASELINE)
    assert "Per-environment delta" in md_text
    log("smoke aggregation + plots + summary OK")

    print()
    print("=" * 60)
    print("=== ALL B3 SMOKE TESTS PASSED ===")
    print("Pipeline plumbing works; safe to run production cells with real DINOv3 + SAM 3.")
    print("=" * 60)
""")

code("""# ===== SMOKE 3: cleanup =====
if RUN_SMOKE_TESTS:
    import shutil
    shutil.rmtree('./_smoke_b3', ignore_errors=True)
    if '_real_load_dinov3' in globals(): globals()['load_dinov3_backbone'] = _real_load_dinov3
    if '_real_sam3_cls'   in globals(): globals()['Sam3ConceptMasker']    = _real_sam3_cls
    log("smoke artifacts removed; production caches untouched; DINOv3/SAM3 stubs restored")
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

**Loading order:** DINOv3 + SAM 3 are loaded once at the start of PROD 1 and
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
