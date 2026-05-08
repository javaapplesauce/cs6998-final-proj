"""Execute baseline_3_dinov3_sam3.ipynb cells in order with stubs.

Same approach as run_notebook_offline.py: extract every code cell, exec() in a
single namespace, with DINOv3 + SAM 3 + sim deps stubbed before any cell runs.
Skips PROD cells (which need real gated weights).
"""
import sys, json, types, torch, torch.nn as nn, pickle
from pathlib import Path

# ---- stub broken pyarrow on this Mac ----
import pandas as pd
class _Tbl:
    def __init__(self, df): self._df = df
    def to_pandas(self): return self._df
def _fake_read_table(path): return _Tbl(pd.read_pickle(str(path)))
def _to_pickle_under_parquet_name(self, p, *a, **kw): return self.to_pickle(str(p))
import importlib.machinery as _machinery
_fake_pq = types.ModuleType('pyarrow.parquet'); _fake_pq.read_table = _fake_read_table
_fake_pa = types.ModuleType('pyarrow'); _fake_pa.parquet = _fake_pq
_fake_pa.__spec__ = _machinery.ModuleSpec('pyarrow', None)
_fake_pq.__spec__ = _machinery.ModuleSpec('pyarrow.parquet', None)
sys.modules['pyarrow'] = _fake_pa
sys.modules['pyarrow.parquet'] = _fake_pq
pd.DataFrame.to_parquet = _to_pickle_under_parquet_name
pd.read_parquet = pd.read_pickle

# ---- stub torch.hub.load for DINOv3 ----
# Production ViT-B/16 uses patch=16, so feature_res = image_size/16.
class _StubDinov3HubModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 768, kernel_size=16, stride=16)
        self._bulk = nn.Parameter(torch.zeros(1024, 1024))
    def get_intermediate_layers(self, x, *, n=1, reshape=True,
                                  return_class_token=False, norm=True):
        return [self.proj(x)]

_orig_hub_load = torch.hub.load
def _patched_hub_load(*args, **kwargs):
    if args and 'dinov3' in str(args[0]).lower():
        return _StubDinov3HubModel()
    return _orig_hub_load(*args, **kwargs)
torch.hub.load = _patched_hub_load

# ---- stub sam3 module entirely ----
fake_sam3       = types.ModuleType('sam3')
fake_builder    = types.ModuleType('sam3.model_builder')
fake_processor  = types.ModuleType('sam3.model.sam3_image_processor')
fake_model      = types.ModuleType('sam3.model')

class _StubSam3Model: pass
class _StubSam3Processor:
    def __init__(self, model): self.model = model
    def set_image(self, image): return {'_image_shape': image.size if hasattr(image, 'size') else None}
    def set_text_prompt(self, state, prompt):
        # Return a single fake instance mask shaped (1, H, W) bool, centered Gaussian
        import numpy as np
        H, W = state.get('_image_shape') or (224, 224)
        H, W = (W, H)   # PIL.size returns (W, H)
        cy, cx = H // 2, W // 2
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        blob = (np.exp(-((yy - cy)**2 + (xx - cx)**2) / max(1, (H * W)//4)) > 0.3)
        return {'masks': blob[None], 'boxes': None, 'scores': None}

fake_builder.build_sam3_image_model = lambda checkpoint=None: _StubSam3Model()
fake_processor.Sam3Processor = _StubSam3Processor
fake_sam3.__spec__       = _machinery.ModuleSpec('sam3', None)
fake_builder.__spec__    = _machinery.ModuleSpec('sam3.model_builder', None)
fake_model.__spec__      = _machinery.ModuleSpec('sam3.model', None)
fake_processor.__spec__  = _machinery.ModuleSpec('sam3.model.sam3_image_processor', None)
sys.modules['sam3'] = fake_sam3
sys.modules['sam3.model_builder'] = fake_builder
sys.modules['sam3.model'] = fake_model
sys.modules['sam3.model.sam3_image_processor'] = fake_processor
fake_sam3.model_builder = fake_builder
fake_sam3.model = fake_model
fake_model.sam3_image_processor = fake_processor

# ---- stub subprocess.run for any in-cell git clones ----
import subprocess as _sp
_orig_run = _sp.run
def _patched_run(args, *a, **kw):
    if args and 'git' in str(args[0]) and 'clone' in (args[1] if len(args) > 1 else ''):
        print(f"[stubbed] git clone: {args}")
        if len(args) >= 4: Path(args[-1]).mkdir(parents=True, exist_ok=True)
        return _sp.CompletedProcess(args, 0)
    return _orig_run(args, *a, **kw)
_sp.run = _patched_run

# ---- read notebook ----
nb = json.load(open('baseline_3_dinov3_sam3.ipynb'))
cells = nb['cells']
print(f"loaded {len(cells)} cells")

# Find smoke cleanup cell — that's our stop point (skip PROD which needs gated weights)
smoke_cleanup_idx = None
for i, c in enumerate(cells):
    if c['cell_type'] == 'code' and 'SMOKE 3' in ''.join(c['source']):
        smoke_cleanup_idx = i; break
assert smoke_cleanup_idx is not None, "couldn't find SMOKE 3 cell"
print(f"smoke ends at cell {smoke_cleanup_idx + 1}; PROD cells skipped\n")

ns = {'__name__': '__main__'}

for i, c in enumerate(cells[:smoke_cleanup_idx + 1]):
    if c['cell_type'] != 'code': continue
    src = ''.join(c['source'])
    label = src.splitlines()[0].strip()[:80]
    print(f">>> EXECUTING CELL {i + 1}: {label}")
    try:
        exec(compile(src, f'<cell {i+1}>', 'exec'), ns)
    except Exception as e:
        print(f"!!! CELL {i+1} FAILED: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

print("\n=== ALL B3 EXECUTED CELLS PASSED ===")
