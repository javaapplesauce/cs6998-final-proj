"""Smoke-test preprocess_ego4d against synthetic Ego4D-shaped inputs.

Generates:
  - 2 fake mp4 videos of 30s each at 10 fps
  - a narrations.json mimicking the Ego4D schema with 12 narrations across the 2 videos

Runs preprocess_ego4d (with cv2 fallback since decord wheels don't exist for arm64 Mac).
Verifies:
  - output dir contains <clip_uid>/{0..4}.jpg per processed clip
  - manifest.parquet has the right columns and is consistent with disk
  - Ego4DR3MDataset (the actual class from the notebook) loads it cleanly
  - re-running with skip_existing=True is a no-op
"""
import os, sys, json, time, random, shutil, subprocess, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image

# ---- stub pyarrow + decord so we can run on this broken-pyarrow Mac ----
import types
class _Tbl:
    def __init__(self, df): self._df = df
    def to_pandas(self): return self._df
def _read_table(path): return _Tbl(pd.read_pickle(str(path)))
def _to_pickle_under_parquet_name(self, p, *a, **kw): return self.to_pickle(str(p))
fake_pq = types.ModuleType('pyarrow.parquet'); fake_pq.read_table = _read_table
fake_pa = types.ModuleType('pyarrow'); fake_pa.parquet = fake_pq
sys.modules['pyarrow'] = fake_pa; sys.modules['pyarrow.parquet'] = fake_pq
pd.DataFrame.to_parquet = _to_pickle_under_parquet_name
pd.read_parquet = pd.read_pickle

# ============================================================
# preprocess_ego4d  (verbatim mirror of the notebook cell, with cv2 fallback)
# ============================================================
def _open_video(path):
    """Returns (reader_obj, fps, n_frames, get_frame_fn)."""
    try:
        import decord
        decord.bridge.set_bridge('native')
        vr = decord.VideoReader(str(path))
        fps = float(vr.get_avg_fps()); n = len(vr)
        def get_frames(indices):
            return vr.get_batch(list(indices)).asnumpy()  # (N,H,W,3) uint8 RGB
        return vr, fps, n, get_frames
    except ImportError:
        # OpenCV fallback (slower; sequential seek)
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
                out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))   # cv2 returns BGR
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
                    reader, fps, n_video_frames, get_frames = _open_video(cands[0])
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
                print(f"  [{i+1:,}/{len(flat):,}] done={n_done:,} skip={n_skip:,} fail={n_fail:,} eta={eta:.1f}min")
    finally:
        if reader is not None: _close_video(reader)

    pd.DataFrame(rows).to_parquet(manifest_path)
    print(f"\n[done] {n_done:,} clips written, {n_skip:,} skipped, {n_fail:,} failed")
    return manifest_path

# ============================================================
# Generate synthetic Ego4D-shaped data
# ============================================================
WORK = Path('./_pp_smoke')
shutil.rmtree(WORK, ignore_errors=True)
videos_dir = WORK / 'videos'; videos_dir.mkdir(parents=True)
out_dir    = WORK / 'frames'

# Build 2 fake mp4 videos: 30s @ 10 fps = 300 frames
def make_video(path, n_frames=300, fps=10, w=160, h=120):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for i in range(n_frames):
        # create a frame whose content varies so we can verify "different frame indices look different"
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :, 0] = (i * 7) % 256
        img[:, :, 1] = (i * 13) % 256
        img[:, :, 2] = (i * 17) % 256
        cv2.putText(img, f"f{i}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        writer.write(img)
    writer.release()

video_uids = ['vid_001', 'vid_002']
for uid in video_uids:
    make_video(videos_dir / f"{uid}.mp4")
print(f"wrote {len(video_uids)} synthetic videos")

# Build a narrations.json mimicking Ego4D schema. Mix #C (kept), #O (filtered), and a no-prefix.
narrations_json = {
    'vid_001': {
        'status': 'complete',
        'narration_pass_1': {
            'narrations': [
                {'timestamp_sec': 5.0,  'timestamp_frame': 50,  'narration_text': '#C C picks up the cup',          'annotation_uid': 'a01'},
                {'timestamp_sec': 12.5, 'timestamp_frame': 125, 'narration_text': '#C C puts the cup on the table', 'annotation_uid': 'a02'},
                {'timestamp_sec': 18.0, 'timestamp_frame': 180, 'narration_text': '#O the man walks by',            'annotation_uid': 'a03'},
                {'timestamp_sec': 22.0, 'timestamp_frame': 220, 'narration_text': '#C C wipes the table',           'annotation_uid': 'a04'},
                {'timestamp_sec': 28.0, 'timestamp_frame': 280, 'narration_text': '#C C closes the door',           'annotation_uid': 'a05'},
                {'timestamp_sec': 0.5,  'timestamp_frame': 5,   'narration_text': '#C C opens the drawer',          'annotation_uid': 'a06'},
            ],
            'summaries': [],
        },
    },
    'vid_002': {
        'status': 'complete',
        'narration_pass_1': {
            'narrations': [
                {'timestamp_sec': 3.0,  'timestamp_frame': 30,  'narration_text': '#C C grabs an apple',         'annotation_uid': 'b01'},
                {'timestamp_sec': 9.0,  'timestamp_frame': 90,  'narration_text': '#C C places apple on plate', 'annotation_uid': 'b02'},
                {'timestamp_sec': 15.0, 'timestamp_frame': 150, 'narration_text': '#C C cuts the apple',        'annotation_uid': 'b03'},
                {'timestamp_sec': 20.0, 'timestamp_frame': 200, 'narration_text': '#C C takes a bite',          'annotation_uid': 'b04'},
                {'timestamp_sec': 25.0, 'timestamp_frame': 250, 'narration_text': '#O another person enters',   'annotation_uid': 'b05'},
                {'timestamp_sec': 29.0, 'timestamp_frame': 290, 'narration_text': '#C C places dish in sink',   'annotation_uid': 'b06'},
            ],
            'summaries': [],
        },
    },
}
narrations_path = WORK / 'narrations.json'
narrations_path.write_text(json.dumps(narrations_json))

# ============================================================
# Run preprocessing
# ============================================================
print("\n=== run 1 (cold) ===")
random.seed(0)
manifest = preprocess_ego4d(
    videos_dir=videos_dir, narrations_json=narrations_path, output_dir=out_dir,
    image_size=84, clip_window_sec=2.0, log_every=4,
)

# ============================================================
# Verify manifest + frames on disk
# ============================================================
m = pd.read_parquet(manifest)
print(f"\nmanifest rows: {len(m)}")

# Expected: 10 #C narrations across the 2 videos (4 #C in vid_001 + 5 #C in vid_002 = ?)
# Counting #C: vid_001 has a01, a02, a04, a05, a06 (5 #C) + a03 #O filtered.
#              vid_002 has b01, b02, b03, b04, b06 (5 #C) + b05 #O filtered.
# Some clips might fail because their window doesn't have 5 frames (vid_001 a06 at t=0.5s, window=2s,
# so f_lo=0, f_hi=25. n_clip=26 >= 5 => OK).
expected_clips = 5 + 5
assert len(m) >= expected_clips - 2, f"got only {len(m)} clips (expected ~{expected_clips})"
assert set(['clip_uid', 'narration', 'n_frames']).issubset(m.columns)
assert (m['n_frames'] == 5).all(), m['n_frames'].unique()
print(f"  [ok] manifest contract holds")

# Filtering check: no #O narrations made it through
assert not m['narration'].str.startswith('#O').any(), "found #O narrations — filter failed"
print("  [ok] #O filter works")

# Frames on disk
for _, row in m.iterrows():
    cdir = out_dir / row['clip_uid']
    assert cdir.is_dir(), f"missing dir {cdir}"
    for j in range(5):
        f = cdir / f"{j}.jpg"
        assert f.exists(), f"missing frame {f}"
        im = Image.open(f); assert im.size == (84, 84), f"wrong size {im.size}"
print(f"  [ok] {len(m)} clips × 5 frames present on disk @ 84×84")

# Frames within a clip should be DIFFERENT (smoke check on indexing):
sample_uid = m.iloc[0]['clip_uid']
arrs = [np.asarray(Image.open(out_dir / sample_uid / f"{j}.jpg")) for j in range(5)]
diffs = [np.abs(arrs[j].astype(int) - arrs[j+1].astype(int)).mean() for j in range(4)]
assert all(d > 1.0 for d in diffs), f"frames suspiciously identical: {diffs}"
print(f"  [ok] frames within a clip are distinct (per-pair mean diff: {[f'{d:.1f}' for d in diffs]})")

# ============================================================
# Verify Ego4DR3MDataset (from the actual notebook) loads the output
# ============================================================
import json as _json
nb = _json.load(open('dinov2_r3m_benchmark.ipynb'))
ego4d_cell_src = None
for c in nb['cells']:
    if c['cell_type'] == 'code' and 'class Ego4DR3MDataset' in ''.join(c['source']):
        ego4d_cell_src = ''.join(c['source']); break
assert ego4d_cell_src is not None, "couldn't find Ego4DR3MDataset cell"

# Stub out things the cell expects
ns = {'__name__': 'smoke',
      'Path': Path, 'random': random, 'torch': __import__('torch'),
      'Image': Image, 'np': np, 'pd': pd}
ns['T'] = __import__('torchvision').transforms
ns['Dataset'] = __import__('torch').utils.data.Dataset
ns['DataLoader']= __import__('torch').utils.data.DataLoader
ns['log'] = lambda msg: print(f"  [log] {msg}")
# We need pq from pyarrow — already stubbed at module level
import pyarrow.parquet as pq; ns['pq'] = pq
exec(compile(ego4d_cell_src, '<ego4d_cell>', 'exec'), ns)

ds = ns['Ego4DR3MDataset'](str(out_dir), 'manifest.parquet', image_size=84, train=False)
print(f"\n[notebook integration] Ego4DR3MDataset loaded: len={len(ds)}")
sample = ds[0]
assert sample['frames'].shape == (5, 3, 84, 84), sample['frames'].shape
print(f"  [ok] sample frames shape {tuple(sample['frames'].shape)}")
print(f"  [ok] sample narration {sample['narration']!r}")

# ============================================================
# Idempotency: re-running should be a no-op
# ============================================================
print("\n=== run 2 (warm — should skip everything) ===")
m_before = pd.read_parquet(manifest).copy()
random.seed(1)
preprocess_ego4d(
    videos_dir=videos_dir, narrations_json=narrations_path, output_dir=out_dir,
    image_size=84, clip_window_sec=2.0, log_every=4,
)
m_after = pd.read_parquet(manifest)
assert len(m_after) == len(m_before)
assert set(m_after['clip_uid']) == set(m_before['clip_uid']), "idempotency broken"
print(f"  [ok] idempotent re-run: {len(m_after)} clips unchanged")

# ============================================================
# cleanup
# ============================================================
shutil.rmtree(WORK)
print("\n=== ALL PREPROCESSING SMOKE TESTS PASSED ===")
