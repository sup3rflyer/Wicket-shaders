#!/usr/bin/env python3
"""Coherency metrics over a captured frame sequence.

Emits a one-line summary and a JSON blob. Hard-fails (exit 2) on *degeneracy*
— non-finite, all-black, or spatially-uniform output across the whole sequence
— which is what a gross STORAGE/SSBO break looks like. The temporal signature
(mean abs frame-to-frame delta on a static source) is REPORTED, not thresholded
here: it is compared against a same-harness Metal reference, because absolute
magnitudes differ across platforms (fp16-RTZ vs fp32, per-tick grain RNG).

Usage: frame_metrics.py <dir> <label>
"""
import sys, os, json, glob
import numpy as np
from PIL import Image

def load_luma(path):
    a = np.asarray(Image.open(path).convert('L'), dtype=np.float64) / 255.0
    return a

def main():
    d, label = sys.argv[1], sys.argv[2]
    paths = sorted(glob.glob(os.path.join(d, '*.png')))
    if not paths:
        print(f"{label}: NO FRAMES CAPTURED", flush=True)
        sys.exit(2)
    frames = [load_luma(p) for p in paths]
    finite = all(np.isfinite(f).all() for f in frames)
    spatial_std = float(np.mean([f.std() for f in frames]))   # structure present?
    frame_mean  = float(np.mean([f.mean() for f in frames]))
    # temporal: mean abs delta between consecutive frames
    deltas = [float(np.mean(np.abs(frames[k] - frames[k-1]))) for k in range(1, len(frames))]
    temporal_mad = float(np.mean(deltas)) if deltas else 0.0
    temporal_max = float(np.max(deltas)) if deltas else 0.0

    degenerate = (not finite) or (frame_mean < 1e-4) or (spatial_std < 1e-4)
    out = dict(label=label, n=len(frames), finite=finite,
               frame_mean=round(frame_mean, 6), spatial_std=round(spatial_std, 6),
               temporal_mad=round(temporal_mad, 6), temporal_max=round(temporal_max, 6),
               degenerate=bool(degenerate))
    print(f"{label:28s} n={out['n']:>3} mean={out['frame_mean']:.4f} "
          f"sstd={out['spatial_std']:.4f} tMAD={out['temporal_mad']:.5f} "
          f"tMax={out['temporal_max']:.5f} finite={finite} "
          f"{'DEGENERATE' if degenerate else 'ok'}", flush=True)
    print("JSON " + json.dumps(out), flush=True)
    sys.exit(2 if degenerate else 0)

if __name__ == '__main__':
    main()
