"""Execute the notebook cells in order with DINOv2/sim deps stubbed,
to verify the smoke-test section integrates cleanly with the definitions.

Approach: extract every code cell, concatenate, and exec() in a single namespace.
Before running, stub:
  - torch.hub.load (returns a small backbone that mimics DINOv2 ViT-B/14 output dim)
  - The CONFIG path overrides happen naturally inside SMOKE_CONFIG.

We skip the PROD cells (5 of them at the end) because they hit real Ego4D paths.
"""
import sys, json, types, torch, torch.nn as nn, pickle
from pathlib import Path

# ---- stub pyarrow.parquet (broken install on this Mac; Linux/A100 box has it) ----
# We back the stub with pandas read/write of a pickle file at the same path.
import pandas as pd

class _FakeTable:
    def __init__(self, df): self._df = df
    def to_pandas(self): return self._df

def _fake_read_table(path):
    return _FakeTable(pd.read_pickle(str(path)))

_fake_pq = types.ModuleType('pyarrow.parquet'); _fake_pq.read_table = _fake_read_table
_fake_pa = types.ModuleType('pyarrow'); _fake_pa.parquet = _fake_pq
# Give stubs a __spec__ so importlib.util.find_spec() works on them
import importlib.machinery as _machinery
_fake_pa.__spec__ = _machinery.ModuleSpec('pyarrow', None)
_fake_pq.__spec__ = _machinery.ModuleSpec('pyarrow.parquet', None)
sys.modules['pyarrow'] = _fake_pa
sys.modules['pyarrow.parquet'] = _fake_pq

# Patch pandas.DataFrame.to_parquet (used by smoke setup) to instead write pickle
def _to_pickle_under_parquet_name(self, path, *a, **kw):
    return self.to_pickle(str(path))
pd.DataFrame.to_parquet = _to_pickle_under_parquet_name

# ---- stub torch.hub.load BEFORE anything imports it ----
class FakeDinov2(nn.Module):
    """Mimics dinov2_vitb14: forward(x) returns CLS-like (B, 768)."""
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_FakeBlock() for _ in range(12)])
        self.norm = nn.LayerNorm(768)
        # Add a few large frozen params so n_frozen > 80M (mimics ViT-B count)
        self._bulk = nn.Parameter(torch.randn(80_500_000 // 1000, 1000) * 0.01)
    def forward(self, x):
        # x: (B, 3, H, W) — just mean-pool to (B, 768) via projection
        feat = x.mean(dim=(2, 3))    # (B, 3)
        feat = nn.functional.pad(feat, (0, 768 - 3))
        for blk in self.blocks: feat = blk(feat)
        return self.norm(feat)

class _FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _FakeAttn()
        self.mlp  = _FakeMlp()
    def forward(self, x):
        return x + self.mlp(x + self.attn(x))

class _FakeAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv  = nn.Linear(768, 3 * 768)
        self.proj = nn.Linear(768, 768)
    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        return self.proj(q + k + v) / 3

class _FakeMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 3072)
        self.fc2 = nn.Linear(3072, 768)
    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


_orig_hub_load = torch.hub.load
def _patched_hub_load(*args, **kwargs):
    if args and 'dinov2' in str(args[0]).lower():
        return FakeDinov2()
    return _orig_hub_load(*args, **kwargs)
torch.hub.load = _patched_hub_load

# ---- stub subprocess.run so the P0 git-clone-r3m-reference cell doesn't actually clone ----
import subprocess as _sp_real
_orig_run = _sp_real.run
def _patched_run(args, *a, **kw):
    if args and 'git' in str(args[0]) and 'clone' in (args[1] if len(args) > 1 else ''):
        print(f"[stubbed] git clone: {args}")
        # Create empty target dir so the existence check is satisfied
        if len(args) >= 4: Path(args[-1]).mkdir(parents=True, exist_ok=True)
        return _sp_real.CompletedProcess(args, 0)
    return _orig_run(args, *a, **kw)
_sp_real.run = _patched_run

# ---- Read notebook, extract code cells ----
nb = json.load(open('baseline_2_dinov3_lora.ipynb'))
cells = nb['cells']
N = len(cells)
print(f"loaded {N} cells")

# Skip the PROD cells (last 6 code cells); we run setup + smoke only.
# Find the smoke cleanup cell index
smoke_cleanup_idx = None
for i, c in enumerate(cells):
    if c['cell_type'] == 'code' and 'SMOKE 3' in ''.join(c['source']):
        smoke_cleanup_idx = i
        break
assert smoke_cleanup_idx is not None, "couldn't find smoke cleanup cell"
print(f"smoke ends at cell {smoke_cleanup_idx + 1}; PROD cells {smoke_cleanup_idx + 3}..{N} skipped")

ns = {'__name__': '__main__'}

for i, c in enumerate(cells[:smoke_cleanup_idx + 1]):
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c['source'])
    label = src.splitlines()[0].strip()[:80]
    print(f"\n>>> EXECUTING CELL {i + 1}: {label}")
    try:
        exec(compile(src, f'<cell {i+1}>', 'exec'), ns)
    except Exception as e:
        print(f"!!! CELL {i+1} FAILED: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

print("\n=== ALL EXECUTED CELLS PASSED (incl. smoke section) ===")
