"""Smoke-test as much of dinov2_r3m_benchmark.ipynb as we can on CPU/Mac.

Exercises (without GPU, without Ego4D, without sim envs):
  - Cell 3:  CONFIG + apply_cuts
  - Cell 4:  Results logger
  - Cell 5:  R3M_BASELINE table
  - Cell 6:  Ego4DR3MDataset against a synthetic in-memory manifest + frames
  - Cell 7:  CLIP text cache build (small subset)
  - Cell 8:  LoRA injection on a toy module
  - Cell 9:  Model assembly (skip torch.hub — replace backbone with a stub)
  - Cell 10: All three loss functions on synthetic embeddings
  - Cell 11: One pretrain step end-to-end with stub backbone
  - Cell 13: encode_images + extract_features helpers
  - Cell 14: BCPolicy forward
  - Cell 16: BC training inner-loop with mocked rollout
  - Cell 18: summarize / delta_table / plot_*  with synthetic sweep
  - Cell 19: make_summary_md

What is intentionally NOT tested here (must be confirmed at runtime on the A100 box):
  - torch.hub.load('facebookresearch/dinov2', ...)  — needs internet + CUDA
  - torch.compile                                   — CUDA-only typically
  - bf16 autocast on CUDA path                      — CPU autocast works differently
  - Real Ego4D dataset I/O                          — depends on pre-extracted cache
  - mj_envs / metaworld / Adroit rollouts           — heavyweight sim deps
"""
import os, sys, json, time, math, random, hashlib, pickle, shutil, types
from pathlib import Path

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device('cpu')          # force CPU for smoke test
DTYPE  = torch.float32

# ============================================================
# Cell 3 — CONFIG  (verbatim except force CPU-friendly knobs)
# ============================================================
CONFIG = {
    'ego4d_root': './_smoke/ego4d', 'manifest_file': 'manifest.parquet',
    'demo_root': './_smoke/demos', 'cache_dir': './_smoke/cache',
    'results_dir': './_smoke/results', 'pretrain_ckpt': './_smoke/cache/pretrain.pt',
    'dinov2_variant': 'dinov2_vitb14', 'clip_variant': 'ViT-B-32-quickgelu',
    'clip_pretrained': 'openai',
    'feat_dim': 32, 'text_dim': 16,                  # tiny dims for smoke
    'lora_rank': 4, 'lora_alpha': 4, 'lora_dropout': 0.0,
    'lora_targets': ('qkv', 'proj', 'fc1', 'fc2'),
    'lang_head_hidden': (32, 32, 32, 32),
    'pretrain_steps': 4, 'pretrain_lr': 1e-3, 'pretrain_weight_decay': 0.0,
    'batch_clips': 4, 'frames_per_clip': 5, 'image_size': 32,
    'num_negatives': 2,
    'lambda_tcn': 1.0, 'lambda_lang': 1.0, 'lambda_l1': 1e-3, 'lambda_l2': 1e-3,
    'pretrain_log_every': 2, 'pretrain_save_every': 1000,
    'precision': 'fp32', 'compile_model': False, 'compile_mode': 'reduce-overhead',
    'num_workers': 0, 'prefetch_factor': 2, 'persistent_workers': False, 'pin_memory': False,
    'bc_lr': 1e-3, 'bc_batch': 8, 'bc_steps': 100, 'bc_eval_every': 25,
    'bc_hidden': (32, 32), 'bc_eval_episodes': 2, 'bc_early_stop_patience': 3,
    'envs': {
        'franka_kitchen': {'tasks': ['slide'], 'views': ['view1'],
                           'demo_sizes': [5], 'horizon': 5},
        'metaworld':      {'tasks': ['assembly-v2'], 'views': ['view1'],
                           'demo_sizes': [5], 'horizon': 5},
    },
    'seeds': [0], 'cut_level': 0,
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
shutil.rmtree('./_smoke', ignore_errors=True)
Path(CONFIG['cache_dir']).mkdir(parents=True, exist_ok=True)
Path(CONFIG['results_dir']).mkdir(parents=True, exist_ok=True)

print("=== CELL 3 (CONFIG)  PASS")

# ============================================================
# Cell 4 — Results logger
# ============================================================
RESULTS_PATH = Path(CONFIG['results_dir']) / 'results.json'

class Results:
    def __init__(self, path):
        self.path = Path(path)
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        else:
            self.data = {'config': CONFIG, 'pretrain': {}, 'bc': {}, 'meta': {'started_at': time.time()}}
            self._flush()
    def log(self, section, key, value):
        self.data.setdefault(section, {})[key] = value; self._flush()
    def append(self, section, key, value):
        self.data.setdefault(section, {}).setdefault(key, []).append(value); self._flush()
    def _flush(self):
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.data, indent=2, default=str)); tmp.replace(self.path)

results = Results(RESULTS_PATH)
results.log('test', 'k', 'v'); results.append('test', 'l', 1); results.append('test', 'l', 2)
assert json.loads(RESULTS_PATH.read_text())['test']['l'] == [1, 2]
print("=== CELL 4 (Results)  PASS")

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ============================================================
# Cell 5 — R3M_BASELINE
# ============================================================
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
print("=== CELL 5 (R3M_BASELINE)  PASS")

# ============================================================
# Cell 6 — Dataset.  Mock parquet via injected pyarrow stub.
# ============================================================
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms as T

# Build synthetic Ego4D-like cache: 8 clips, 5 frames each, narrations
ego4d_root = Path(CONFIG['ego4d_root']); ego4d_root.mkdir(parents=True, exist_ok=True)
narrations = [f"narration {i} doing thing" for i in range(8)]
df_rows = []
for i, n in enumerate(narrations):
    uid = f"clip_{i:03d}"
    (ego4d_root / uid).mkdir(parents=True, exist_ok=True)
    for j in range(5):
        arr = (np.random.rand(48, 48, 3) * 255).astype(np.uint8)
        Image.fromarray(arr).save(ego4d_root / uid / f"{j}.jpg", quality=85)
    df_rows.append({'clip_uid': uid, 'narration': n, 'n_frames': 5})
df = pd.DataFrame(df_rows)

# Stub pyarrow.parquet: read_table().to_pandas() must return our df
fake_pa = types.ModuleType('pyarrow'); fake_pq = types.ModuleType('pyarrow.parquet')
class _Tbl:
    def to_pandas(self): return df
def _read_table(path): assert Path(path).parent.exists(); return _Tbl()
fake_pq.read_table = _read_table
fake_pa.parquet = fake_pq
sys.modules['pyarrow'] = fake_pa
sys.modules['pyarrow.parquet'] = fake_pq
(ego4d_root / 'manifest.parquet').write_text('stub')   # presence check only

import pyarrow.parquet as pq

class Ego4DR3MDataset(Dataset):
    def __init__(self, root, manifest_file, image_size=224, train=True):
        self.root = Path(root); self.image_size = image_size; self.train = train
        manifest_path = self.root / manifest_file
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing {manifest_path}")
        self.df = pq.read_table(manifest_path).to_pandas()
        self.df = self.df[self.df['n_frames'] >= 5].reset_index(drop=True)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    def __len__(self): return len(self.df)
    def _sample_indices(self, n):
        first20 = max(1, n // 5); last20 = max(1, n // 5)
        i0 = random.randint(0, first20 - 1); ig = random.randint(n - last20, n - 1)
        lo, hi = i0 + 1, ig - 1
        if hi - lo + 1 < 3:
            mids = sorted([random.randint(lo, hi) for _ in range(3)]) if hi >= lo else [i0, i0, i0]
        else:
            mids = sorted(random.sample(range(lo, hi + 1), 3))
        return [i0, mids[0], mids[1], mids[2], ig]
    def _load(self, uid, indices):
        imgs = [Image.open(self.root / uid / f"{i}.jpg").convert('RGB') for i in indices]
        S = self.image_size
        imgs = [im.resize((S, S), Image.Resampling.BICUBIC) for im in imgs]
        arr = np.stack([np.asarray(im) for im in imgs], axis=0)
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
        return self.normalize(x)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        indices = [0, 1, 2, 3, 4] if row['n_frames'] == 5 else self._sample_indices(int(row['n_frames']))
        return {'frames': self._load(row['clip_uid'], indices),
                'narration': row['narration'], 'clip_uid': row['clip_uid']}

ds = Ego4DR3MDataset(CONFIG['ego4d_root'], 'manifest.parquet', image_size=CONFIG['image_size'])
assert len(ds) == 8
sample = ds[0]
assert sample['frames'].shape == (5, 3, CONFIG['image_size'], CONFIG['image_size'])
print(f"=== CELL 6 (Dataset)  PASS  (sample shape {tuple(sample['frames'].shape)})")

# Loader collate
def collate(batch):
    return {'frames': torch.stack([b['frames'] for b in batch], dim=0),
            'narration': [b['narration'] for b in batch],
            'clip_uid': [b['clip_uid'] for b in batch]}

loader = DataLoader(ds, batch_size=CONFIG['batch_clips'], shuffle=True, drop_last=True,
                    num_workers=0, collate_fn=collate)
batch = next(iter(loader))
assert batch['frames'].shape == (CONFIG['batch_clips'], 5, 3, CONFIG['image_size'], CONFIG['image_size'])
print("=== CELL 6 (DataLoader collate)  PASS")

# ============================================================
# Cell 7 — CLIP text cache (real open_clip ViT-B/32, downloads weights once)
# ============================================================
import open_clip
CLIP_CACHE = Path(CONFIG['cache_dir']) / 'clip_text_embeds.pt'

@torch.no_grad()
def build_clip_text_cache(cfg, force=False):
    if CLIP_CACHE.exists() and not force:
        return torch.load(CLIP_CACHE, map_location='cpu')
    ds = Ego4DR3MDataset(cfg['ego4d_root'], cfg['manifest_file'], image_size=cfg['image_size'], train=False)
    uniq = sorted(set(ds.df['narration'].astype(str).tolist()))
    clip_model, _, _ = open_clip.create_model_and_transforms(
        cfg['clip_variant'], pretrained=cfg['clip_pretrained'], device=DEVICE)
    clip_model.eval()
    tok = open_clip.get_tokenizer(cfg['clip_variant'])
    embeds = []
    for i in range(0, len(uniq), 64):
        batch = uniq[i:i+64]
        e = clip_model.encode_text(tok(batch).to(DEVICE))
        embeds.append(F.normalize(e.float(), dim=-1).cpu())
    embeds = torch.cat(embeds, dim=0)
    cache = {'narrations': uniq, 'embeds': embeds, 'narration_to_idx': {n: i for i, n in enumerate(uniq)}}
    torch.save(cache, CLIP_CACHE)
    return cache

# Override CONFIG['text_dim'] only for this test (real CLIP returns 512)
CONFIG['text_dim'] = 512
print("downloading CLIP ViT-B/32 (one-time, ~500MB) ...")
CLIP_TEXT_CACHE = build_clip_text_cache(CONFIG)
assert CLIP_TEXT_CACHE['embeds'].shape == (8, 512)
print(f"=== CELL 7 (CLIP cache)  PASS  ({CLIP_TEXT_CACHE['embeds'].shape})")

# ============================================================
# Cell 8 — LoRA injection on a toy ViT-like block (no DINOv2 download)
# ============================================================
class ToyAttn(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        return self.proj(q + k + v) / 3

class ToyMlp(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.fc1 = nn.Linear(d, d * 2); self.fc2 = nn.Linear(d * 2, d)
    def forward(self, x): return self.fc2(F.gelu(self.fc1(x)))

class ToyBlock(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.attn = ToyAttn(d); self.mlp = ToyMlp(d)
    def forward(self, x): return x + self.mlp(x + self.attn(x))

class ToyViT(nn.Module):
    def __init__(self, d=32, depth=4):
        super().__init__()
        self.blocks = nn.ModuleList([ToyBlock(d) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
    def forward(self, x):
        # x: (B, 3, H, W) -> tokens (B, N, d). Toy: just mean-pool to (B, d).
        if x.dim() == 4: x = x.mean(dim=(2, 3))   # (B, 3) -> ignore
        x = nn.functional.pad(x, (0, 32 - x.shape[-1]))
        for blk in self.blocks: x = blk(x)
        return self.norm(x)

class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dropout):
        super().__init__()
        self.base = base
        for p in self.base.parameters(): p.requires_grad = False
        self.r = r; self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    def forward(self, x):
        out = self.base(x)
        x_d = self.dropout(x)
        delta = (x_d @ self.lora_A.t()) @ self.lora_B.t()
        return out + delta * self.scaling

def inject_lora(module, target_substrings, r, alpha, dropout):
    n = 0
    for name, child in list(module.named_modules()):
        for cname, c in list(child.named_children()):
            if isinstance(c, nn.Linear) and any(t in cname for t in target_substrings):
                setattr(child, cname, LoRALinear(c, r, alpha, dropout))
                n += 1
    return n

def count_trainable(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

vit = ToyViT(d=32, depth=4)
for p in vit.parameters(): p.requires_grad = False
n_repl = inject_lora(vit, ('qkv', 'proj', 'fc1', 'fc2'), r=4, alpha=4, dropout=0.0)
expected_modules = 4 * 4   # 4 blocks × {qkv, proj, fc1, fc2}
assert n_repl == expected_modules, f"expected {expected_modules}, got {n_repl}"
trainable = count_trainable(vit)
# Each LoRA: r*in + out*r params; for 16 modules total
assert trainable > 0
y = vit(torch.randn(2, 32))    # forward smoke
assert y.shape == (2, 32)
print(f"=== CELL 8 (LoRA inject)  PASS  ({n_repl} replacements, {trainable} trainable)")

# Test gradient flow
x = torch.randn(2, 32)
y = vit(x).sum()
y.backward()
grads = [p.grad for p in vit.parameters() if p.requires_grad]
assert all(g is not None for g in grads), "some LoRA params got no grad"
assert any(g.abs().sum().item() > 0 for g in grads), "all gradients zero"
print("=== CELL 8 (LoRA gradient flow)  PASS")

# ============================================================
# Cell 9 — Model assembly with stub backbone
# ============================================================
class LanguageHead(nn.Module):
    def __init__(self, feat_dim, text_dim, hidden=(1024, 1024, 1024, 1024)):
        super().__init__()
        in_dim = 2 * feat_dim + text_dim
        layers = []; d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU()]; d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, z0, zj, l):
        return self.net(torch.cat([z0, zj, l], dim=-1)).squeeze(-1)

class StubBackbone(nn.Module):
    """Stand-in for DINOv2 ViT-B/14 for offline smoke. Returns (B, feat_dim)."""
    def __init__(self, feat_dim=32):
        super().__init__()
        self.proj = nn.Linear(3, feat_dim)
        self.attn = ToyAttn(d=feat_dim); self.mlp = ToyMlp(d=feat_dim)
    def forward(self, x):
        # x: (B, 3, H, W) -> mean-pool patches -> linear -> attn -> mlp
        feat = self.proj(x.mean(dim=(2, 3)))
        feat = self.attn(feat); feat = self.mlp(feat)
        return feat

class DinoLoRAR3M(nn.Module):
    def __init__(self, cfg, backbone=None):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone if backbone is not None else StubBackbone(cfg['feat_dim'])
        for p in self.backbone.parameters(): p.requires_grad = False
        n = inject_lora(self.backbone, cfg['lora_targets'],
                        r=cfg['lora_rank'], alpha=cfg['lora_alpha'], dropout=cfg['lora_dropout'])
        self.lang_head = LanguageHead(cfg['feat_dim'], cfg['text_dim'], cfg['lang_head_hidden'])
        print(f"  injected lora into {n} modules")
    def encode(self, x): return self.backbone(x)
    def forward_pretrain(self, frames):
        B, K, C, H, W = frames.shape
        z = self.encode(frames.view(B * K, C, H, W)).view(B, K, -1)
        z0, zi, zj, zk, zg = z.unbind(dim=1)
        return {'z0': z0, 'zi': zi, 'zj': zj, 'zk': zk, 'zg': zg}
    def lang_score(self, z0, zt, l): return self.lang_head(z0, zt, l)

model = DinoLoRAR3M(CONFIG)
out = model.forward_pretrain(batch['frames'])
for k in ('z0', 'zi', 'zj', 'zk', 'zg'):
    assert out[k].shape == (CONFIG['batch_clips'], CONFIG['feat_dim'])
print("=== CELL 9 (Model assembly)  PASS")

# ============================================================
# Cell 10 — Loss functions
# ============================================================
def neg_l2(a, b): return -((a - b).pow(2).sum(dim=-1))

def time_contrastive_loss(out, num_neg=3):
    zi, zj, zk = out['zi'], out['zj'], out['zk']
    B = zi.shape[0]
    rand_idx = torch.stack([
        torch.tensor(random.sample([j for j in range(B) if j != b], k=min(num_neg, B - 1)), device=zi.device)
        for b in range(B)
    ], dim=0)
    s_pos     = neg_l2(zi, zj).unsqueeze(1)
    s_far     = neg_l2(zi, zk).unsqueeze(1)
    zi_neg    = zi[rand_idx]
    s_crossvid= neg_l2(zi.unsqueeze(1).expand_as(zi_neg), zi_neg)
    logits = torch.cat([s_pos, s_far, s_crossvid], dim=1)
    return F.cross_entropy(logits, torch.zeros(B, dtype=torch.long, device=zi.device))

def video_language_loss(model, out, l_embed, num_neg=3):
    z0, zi, zj, zk, zg = out['z0'], out['zi'], out['zj'], out['zk'], out['zg']
    B = z0.shape[0]
    perm = (torch.arange(B, device=l_embed.device) + 1) % B    # cyclic shift derangement
    l_wrong = l_embed[perm]
    losses = []
    for z_pos, z_neg in [(zg, z0), (zj, zi), (zk, zj)]:
        s_pos = model.lang_score(z0, z_pos, l_embed)
        s_nv  = model.lang_score(z0, z_neg, l_embed)
        s_nl  = model.lang_score(z0, z_pos, l_wrong)
        logits = torch.stack([s_pos, s_nv, s_nl], dim=1)
        losses.append(F.cross_entropy(logits, torch.zeros(B, dtype=torch.long, device=z0.device)))
    return sum(losses) / len(losses)

def regularization(out, lam_l1, lam_l2):
    z = torch.stack([out['zi'], out['zj'], out['zk']], dim=1)
    return lam_l1 * z.abs().mean() + lam_l2 * z.pow(2).mean()

l_emb = torch.randn(CONFIG['batch_clips'], CONFIG['text_dim'])
l_tcn = time_contrastive_loss(out, num_neg=CONFIG['num_negatives'])
l_lng = video_language_loss(model, out, l_emb, num_neg=CONFIG['num_negatives'])
l_reg = regularization(out, CONFIG['lambda_l1'], CONFIG['lambda_l2'])
total = l_tcn + l_lng + l_reg
total.backward()
assert torch.isfinite(l_tcn) and torch.isfinite(l_lng) and torch.isfinite(l_reg)
print(f"=== CELL 10 (Losses)  PASS  tcn={l_tcn.item():.3f} lang={l_lng.item():.3f} reg={l_reg.item():.6f}")

# ============================================================
# Cell 11 — pretraining loop (4 steps)
# ============================================================
def lookup_text_embeds(narrations, text_cache):
    idx = torch.tensor([text_cache['narration_to_idx'][n] for n in narrations], dtype=torch.long)
    return text_cache['embeds'][idx]

def pretrain(cfg, model, opt, loader, text_cache, save_path, max_steps=None):
    max_steps = max_steps or cfg['pretrain_steps']
    model.train()
    step = 0
    iterator = iter(loader)
    while step < max_steps:
        try: batch = next(iterator)
        except StopIteration:
            iterator = iter(loader); batch = next(iterator)
        frames = batch['frames']
        l_embeds = lookup_text_embeds(batch['narration'], text_cache)
        opt.zero_grad(set_to_none=True)
        out = model.forward_pretrain(frames)
        l_tcn = time_contrastive_loss(out, num_neg=cfg['num_negatives'])
        l_lng = video_language_loss(model, out, l_embeds, num_neg=cfg['num_negatives'])
        l_reg = regularization(out, cfg['lambda_l1'], cfg['lambda_l2'])
        total = cfg['lambda_tcn'] * l_tcn + cfg['lambda_lang'] * l_lng + l_reg
        total.backward(); opt.step()
        step += 1
    torch.save({'step': step,
                'state_dict': {k: v for k, v in model.state_dict().items()
                               if 'lora_' in k or 'lang_head' in k},
                'config': cfg}, save_path)

# Recreate model + opt for clean state
torch.manual_seed(0)
model = DinoLoRAR3M(CONFIG)
trainable = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.Adam(trainable, lr=CONFIG['pretrain_lr'])
pretrain(CONFIG, model, opt, loader, CLIP_TEXT_CACHE, CONFIG['pretrain_ckpt'])
assert Path(CONFIG['pretrain_ckpt']).exists()
ckpt = torch.load(CONFIG['pretrain_ckpt'], weights_only=False)
n_lora_keys = sum(1 for k in ckpt['state_dict'] if 'lora_' in k)
assert n_lora_keys > 0
print(f"=== CELL 11 (pretrain 4 steps)  PASS  ({n_lora_keys} lora keys saved)")

# ============================================================
# Cell 13 — load_pretrained_model + encode_images
# ============================================================
def load_pretrained_model(cfg, ckpt_path):
    model = DinoLoRAR3M(cfg)
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)['state_dict']
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    return model

@torch.no_grad()
def encode_images(model, imgs_uint8, batch=64):
    N = imgs_uint8.shape[0]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    feats = []
    for i in range(0, N, batch):
        chunk = imgs_uint8[i:i + batch]
        if chunk.shape[1] != CONFIG['image_size'] or chunk.shape[2] != CONFIG['image_size']:
            chunk = np.stack([
                np.asarray(Image.fromarray(im).resize((CONFIG['image_size'], CONFIG['image_size']),
                                                       Image.Resampling.BICUBIC))
                for im in chunk], axis=0)
        x = torch.from_numpy(chunk).permute(0, 3, 1, 2).float() / 255.0
        x = (x - mean) / std
        feats.append(model.encode(x).float().cpu())
    return torch.cat(feats, dim=0)

m = load_pretrained_model(CONFIG, CONFIG['pretrain_ckpt'])
imgs = (np.random.rand(7, CONFIG['image_size'], CONFIG['image_size'], 3) * 255).astype(np.uint8)
feats = encode_images(m, imgs, batch=4)
assert feats.shape == (7, CONFIG['feat_dim'])
print(f"=== CELL 13 (load + encode_images)  PASS  ({feats.shape})")

# Build synthetic demos for the BC sweep
demo_root = Path(CONFIG['demo_root'])
for env_name, env_cfg in CONFIG['envs'].items():
    for task in env_cfg['tasks']:
        for view in env_cfg['views']:
            d = demo_root / env_name / task / view; d.mkdir(parents=True, exist_ok=True)
            for i in range(6):
                T_steps = env_cfg.get('horizon', 5) + 1
                pickle.dump({
                    'images':  (np.random.rand(T_steps, 32, 32, 3) * 255).astype(np.uint8),
                    'actions': np.random.randn(T_steps, 4).astype(np.float32),
                    'proprio': np.random.randn(T_steps, 8).astype(np.float32),
                    'success': True,
                }, open(d / f"demo_{i}.pkl", 'wb'))

def extract_features_for_env(cfg, model, env_name, env_cfg):
    out_dir = Path(cfg['cache_dir']) / 'features' / env_name; out_dir.mkdir(parents=True, exist_ok=True)
    for task in env_cfg['tasks']:
        for view in env_cfg['views']:
            cache_file = out_dir / f"{task}__{view}.pt"
            if cache_file.exists(): continue
            demo_glob = Path(cfg['demo_root']) / env_name / task / view
            if not demo_glob.exists(): continue
            episodes = []
            for f in sorted(demo_glob.glob('demo_*.pkl')):
                d = pickle.load(open(f, 'rb'))
                feats = encode_images(model, d['images'])
                episodes.append({'features': feats,
                                 'actions': torch.as_tensor(d['actions'], dtype=torch.float32),
                                 'proprio': torch.as_tensor(d['proprio'], dtype=torch.float32),
                                 'success': bool(d.get('success', True))})
            torch.save({'env': env_name, 'task': task, 'view': view, 'episodes': episodes}, cache_file)

for env_name, env_cfg in CONFIG['envs'].items():
    extract_features_for_env(CONFIG, m, env_name, env_cfg)
print("=== CELL 13 (extract_features_for_env)  PASS")

# ============================================================
# Cell 14 — BC dataset + policy
# ============================================================
class FeatureBCDataset(Dataset):
    def __init__(self, cache_file, n_demos):
        b = torch.load(cache_file, map_location='cpu', weights_only=False)
        eps = b['episodes'][:n_demos]
        self.feats   = torch.cat([e['features'] for e in eps], dim=0)
        self.proprio = torch.cat([e['proprio']  for e in eps], dim=0)
        self.actions = torch.cat([e['actions']  for e in eps], dim=0)
    def __len__(self): return self.feats.shape[0]
    def __getitem__(self, i):
        return torch.cat([self.feats[i], self.proprio[i]], dim=-1), self.actions[i]

class BCPolicy(nn.Module):
    def __init__(self, in_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim)
        layers = []; d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]; d = h
        layers += [nn.Linear(d, act_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(self.bn(x))

cf = Path(CONFIG['cache_dir']) / 'features' / 'franka_kitchen' / 'slide__view1.pt'
bc_ds = FeatureBCDataset(cf, n_demos=5)
in_dim = bc_ds.feats.shape[1] + bc_ds.proprio.shape[1]
policy = BCPolicy(in_dim, bc_ds.actions.shape[1], hidden=CONFIG['bc_hidden'])
xx, aa = next(iter(DataLoader(bc_ds, batch_size=8)))
pred = policy(xx); assert pred.shape == aa.shape
print(f"=== CELL 14 (BC dataset + policy)  PASS  in_dim={in_dim} act_dim={aa.shape[1]}")

# ============================================================
# Cell 16 — BC training inner loop with mocked rollout
# ============================================================
# Mock rollout and env: returns random success
def mock_rollout(env, env_name, encoder, policy, horizon, image_size=224):
    return random.random() > 0.5

def mock_make_env(env_name, task, view, image_size=224):
    return types.SimpleNamespace(close=lambda: None)

def train_bc_one(cfg, env_name, task, view, n_demos, seed, encoder):
    random.seed(seed); torch.manual_seed(seed)
    cache_file = Path(cfg['cache_dir']) / 'features' / env_name / f"{task}__{view}.pt"
    if not cache_file.exists():
        return {'best_success': None, 'reason': 'features missing'}
    ds = FeatureBCDataset(cache_file, n_demos=n_demos)
    in_dim = ds.feats.shape[1] + ds.proprio.shape[1]
    policy = BCPolicy(in_dim, ds.actions.shape[1], hidden=cfg['bc_hidden'])
    opt = torch.optim.Adam(policy.parameters(), lr=cfg['bc_lr'])
    loader_inner = DataLoader(ds, batch_size=cfg['bc_batch'], shuffle=True, drop_last=True)
    horizon = (cfg['envs'][env_name].get('horizons', {}) or {}).get(task,
                cfg['envs'][env_name].get('horizon'))
    env = mock_make_env(env_name, task, view, image_size=cfg['image_size'])
    best = -1.0; plateau = 0; history = []; iterator = iter(loader_inner)
    for step in range(1, cfg['bc_steps'] + 1):
        try: x, a = next(iterator)
        except StopIteration:
            iterator = iter(loader_inner); x, a = next(iterator)
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(policy(x), a)
        loss.backward(); opt.step()
        if step % cfg['bc_eval_every'] == 0:
            policy.eval()
            successes = sum(mock_rollout(env, env_name, encoder, policy, horizon, cfg['image_size'])
                            for _ in range(cfg['bc_eval_episodes']))
            rate = 100.0 * successes / cfg['bc_eval_episodes']
            history.append({'step': step, 'success_pct': rate, 'loss': float(loss.item())})
            if rate > best: best = rate; plateau = 0
            else: plateau += 1
            if plateau >= cfg['bc_early_stop_patience']: break
            policy.train()
    return {'best_success': float(best), 'history': history, 'env': env_name,
            'task': task, 'view': view, 'n_demos': n_demos, 'seed': seed}

r = train_bc_one(CONFIG, 'franka_kitchen', 'slide', 'view1', 5, 0, m)
assert r['best_success'] is not None
assert r['history'], "history should be non-empty"
print(f"=== CELL 16 (train_bc_one mocked)  PASS  best={r['best_success']}  history={len(r['history'])}")

# ============================================================
# Cell 18 — summarize + delta + plots
# ============================================================
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

# Build synthetic SWEEP
SWEEP = []
for env in ['franka_kitchen', 'metaworld']:
    for task in ['slide', 'light'] if env == 'franka_kitchen' else ['assembly-v2', 'hammer-v2']:
        for view in ['view1', 'view2']:
            for nd in [5, 25]:
                for seed in [0, 1]:
                    SWEEP.append({'env': env, 'task': task, 'view': view,
                                  'n_demos': nd, 'seed': seed,
                                  'best_success': 50.0 + random.random() * 30,
                                  'history': []})

def summarize(sweep):
    by_env, by_task, by_view, by_demos = {}, {}, {}, {}
    for r in sweep:
        if r.get('best_success') is None: continue
        by_env.setdefault(r['env'], []).append(r['best_success'])
        by_task.setdefault((r['env'], r['task']), []).append(r['best_success'])
        by_view.setdefault((r['env'], r['view']), []).append(r['best_success'])
        by_demos.setdefault((r['env'], r['n_demos']), []).append(r['best_success'])
    def red(d):
        return {k: {'mean': float(np.mean(v)),
                    'stderr': float(np.std(v) / np.sqrt(len(v))),
                    'n': len(v)} for k, v in d.items()}
    return {'env': red(by_env), 'task': red(by_task), 'view': red(by_view), 'demos': red(by_demos)}

def delta_table(ours, baseline):
    rows = []
    for env in ['franka_kitchen', 'metaworld', 'adroit']:
        if env not in ours['env']: continue
        o = ours['env'][env]['mean']; b = baseline['aggregate'][env]['mean']
        rows.append({'env': env, 'ours_mean': o, 'r3m_mean': b, 'delta': o - b,
                     'ours_stderr': ours['env'][env]['stderr'],
                     'r3m_stderr':  baseline['aggregate'][env]['stderr']})
    return rows

ours = summarize(SWEEP)
delta = delta_table(ours, R3M_BASELINE)
assert len(delta) == 2
assert all('delta' in r for r in delta)
print(f"=== CELL 18 (summarize/delta)  PASS  delta={[(r['env'], round(r['delta'], 2)) for r in delta]}")

def plot_side_by_side(rows, out_path):
    envs = [r['env'] for r in rows]
    ours = [r['ours_mean'] for r in rows]; r3m = [r['r3m_mean'] for r in rows]
    fig, ax = plt.subplots()
    x = np.arange(len(envs)); w = 0.4
    ax.bar(x - w/2, ours, w, yerr=[r['ours_stderr'] for r in rows], capsize=3, label='ours')
    ax.bar(x + w/2, r3m,  w, yerr=[r['r3m_stderr']  for r in rows], capsize=3, label='R3M')
    ax.set_xticks(x); ax.set_xticklabels(envs); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=80); plt.close(fig)

plot_side_by_side(delta, Path(CONFIG['results_dir']) / 'side_by_side.png')
assert (Path(CONFIG['results_dir']) / 'side_by_side.png').exists()
print("=== CELL 18 (plot_side_by_side)  PASS")

# ============================================================
# Cell 19 — markdown summary
# ============================================================
def make_summary_md(ours, delta_rows, baseline):
    lines = []
    lines.append("# DINOv2+LoRA vs R3M — Results")
    lines.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')}_\n")
    lines.append("## Per-environment delta\n")
    lines.append("| Environment | Ours | R3M | Δ |")
    lines.append("|---|---:|---:|---:|")
    for r in delta_rows:
        lines.append(f"| {r['env']} | {r['ours_mean']:.1f} | {r['r3m_mean']:.1f} | {r['delta']:+.1f} |")
    return "\n".join(lines)

md = make_summary_md(ours, delta, R3M_BASELINE)
assert "Per-environment delta" in md
print("=== CELL 19 (markdown summary)  PASS")

# ============================================================
# Cleanup smoke artifacts
# ============================================================
shutil.rmtree('./_smoke', ignore_errors=True)
print("\n=== ALL SMOKE TESTS PASSED ===")
