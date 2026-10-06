"""Execute baseline_3_dinov2_sam2.ipynb cells in order with stubs.

Stubs DINOv2 (torch.hub), SAM 2 (sam2 module), robosuite, and h5py reads.
Skips PROD cells.
"""
import sys, json, types, torch, torch.nn as nn, pickle, importlib.machinery as _m
from pathlib import Path

# ---- stub broken pyarrow on this Mac (not needed by baseline 3 but harmless) ----
import pandas as pd

# ---- stub torch.hub.load for DINOv2 ----
class _StubDinov2HubModel(nn.Module):
    """Fake DINOv2 ViT-B/14: 768-dim, patch 14 -> 518/14=37 patches."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 768, kernel_size=14, stride=14)
        self._bulk = nn.Parameter(torch.zeros(1024, 1024))
    def get_intermediate_layers(self, x, *, n=1, reshape=True,
                                  return_class_token=False, norm=True):
        return [self.proj(x)]

_orig_hub_load = torch.hub.load
def _patched_hub_load(*args, **kwargs):
    if args and 'dinov2' in str(args[0]).lower():
        return _StubDinov2HubModel()
    return _orig_hub_load(*args, **kwargs)
torch.hub.load = _patched_hub_load

# ---- stub sam2 module ----
fake_sam2     = types.ModuleType('sam2')
fake_build    = types.ModuleType('sam2.build_sam')
fake_pred     = types.ModuleType('sam2.sam2_image_predictor')

class _StubSam2Model:
    def __init__(self): pass
class _StubSam2Predictor:
    def __init__(self, model): self.model = model
    def set_image(self, image): self._H, self._W = image.shape[:2]
    def predict(self, point_coords, point_labels, multimask_output=False):
        import numpy as np
        H, W = self._H, self._W
        cy, cx = H // 2, W // 2
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        blob = (np.exp(-((yy - cy)**2 + (xx - cx)**2) / max(1, (H*W)//16)) > 0.3)
        return blob[None], np.array([0.9], dtype=np.float32), None

fake_build.build_sam2 = lambda cfg, ckpt, device=None: _StubSam2Model()
fake_pred.SAM2ImagePredictor = _StubSam2Predictor
for m, name in [(fake_sam2, 'sam2'),
                (fake_build, 'sam2.build_sam'),
                (fake_pred, 'sam2.sam2_image_predictor')]:
    m.__spec__ = _m.ModuleSpec(name, None)
    sys.modules[name] = m
fake_sam2.build_sam = fake_build
fake_sam2.sam2_image_predictor = fake_pred

# ---- stub robosuite ----
fake_robosuite = types.ModuleType('robosuite')
fake_robosuite.__spec__ = _m.ModuleSpec('robosuite', None)
def _fake_make(**kw):
    return types.SimpleNamespace(
        reset=lambda: {},
        step=lambda a: ({}, 0.0, False, {'success': False}),
        close=lambda: None,
        _get_observations=lambda: {'agentview_image': __import__('numpy').zeros((518, 518, 3), 'uint8'),
                                     'robot0_joint_pos': __import__('numpy').zeros(7),
                                     'robot0_eef_pos': __import__('numpy').zeros(3),
                                     'robot0_eef_quat': __import__('numpy').zeros(4)},
    )
fake_robosuite.make = _fake_make
sys.modules['robosuite'] = fake_robosuite

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

# ---- create a small fake HDF5 at the expected path so the conversion cell runs ----
import h5py, tempfile, os, numpy as np
fake_hdf5 = Path('/tmp/_fake_demos_lift.hdf5')
with h5py.File(fake_hdf5, 'w') as f:
    f.attrs['env_name'] = 'Lift'
    f.attrs['action_dim'] = 6
    f.attrs['proprio_dim'] = 14
    f.attrs['image_size'] = 518
    for i in range(6):
        T = 7
        g = f.create_group(f'demo_{i}')
        g.create_dataset('actions', data=np.random.randn(T, 6).astype('float32'))
        g.create_dataset('images',  data=(np.random.rand(T, 3, 518, 518)*255).astype('uint8'))
        g.create_dataset('proprio', data=np.random.randn(T, 14).astype('float32'))

# Also override the CONFIG hdf5_src in-cell post-exec by exposing a fixup namespace value
# (we patch CONFIG after the CONFIG cell runs)

# ---- read notebook ----
nb = json.load(open('baseline_3_dinov2_sam2.ipynb'))
cells = nb['cells']
print(f"loaded {len(cells)} cells")

smoke_cleanup_idx = None
for i, c in enumerate(cells):
    if c['cell_type'] == 'code' and 'SMOKE 3' in ''.join(c['source']):
        smoke_cleanup_idx = i; break
assert smoke_cleanup_idx is not None
print(f"smoke ends at cell {smoke_cleanup_idx + 1}; PROD cells skipped\n")

ns = {'__name__': '__main__'}

for i, c in enumerate(cells[:smoke_cleanup_idx + 1]):
    if c['cell_type'] != 'code': continue
    src = ''.join(c['source'])
    label = src.splitlines()[0].strip()[:80]
    print(f">>> EXECUTING CELL {i + 1}: {label}")
    try:
        exec(compile(src, f'<cell {i+1}>', 'exec'), ns)
        # After CONFIG cell runs, redirect the HDF5 source to our fake file (and demo_root to tmp)
        if i + 1 == 4 and 'CONFIG' in ns:
            ns['CONFIG']['hdf5_src']  = str(fake_hdf5)
            ns['CONFIG']['demo_root'] = '/tmp/_fake_demos_root'
    except Exception as e:
        print(f"!!! CELL {i+1} FAILED: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

print("\n=== ALL B3 EXECUTED CELLS PASSED ===")
