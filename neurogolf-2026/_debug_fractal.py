"""Debug: does fractal_self_tile detector match task001?"""
import json, numpy as np
from pathlib import Path
TASK_DIR = Path(__file__).parent

with open(TASK_DIR / 'task001.json') as f: t = json.load(f)
pairs = t.get('train', [])
print(f"task001: {len(pairs)} train pairs")
for i, p in enumerate(pairs[:3]):
    ig, og = np.array(p['input']), np.array(p['output'])
    print(f"  pair {i}: in {ig.shape} out {og.shape}")

# Re-implement detector in situ
def detect_fractal_self_tile(pairs, H=30, W=30):
    ih, iw = None, None
    for p in pairs:
        ig = np.array(p['input']); og = np.array(p['output'])
        if ih is None:
            ih, iw = ig.shape
        if ig.shape != (ih, iw):
            print(f"  MISMATCH: ig shape {ig.shape} != ({ih},{iw})")
            return None
        if og.shape != (ih*ih, iw*iw):
            print(f"  MISMATCH og: {og.shape} != ({ih*ih},{iw*iw})")
            return None
        for r in range(ih):
            for c in range(iw):
                tile = og[r*ih:(r+1)*ih, c*iw:(c+1)*iw]
                expected = ig if ig[r,c] != 0 else np.zeros_like(ig)
                if not np.array_equal(tile, expected):
                    print(f"  MISMATCH tile at ({r},{c})")
                    return None
    if ih is None or ih*ih > H or iw*iw > W:
        print(f"  too big: {ih*ih}x{iw*iw}")
        return None
    return ih, iw

r = detect_fractal_self_tile(pairs)
print(f"detect result: {r}")

# Try also on full pairs including test + arc-gen
all_p = pairs + t.get('test', []) + t.get('arc-gen', [])
print(f"\nFull task001 has {len(all_p)} total pairs")
# Check shapes distribution
from collections import Counter
shape_counts = Counter((tuple(np.array(p['input']).shape), tuple(np.array(p['output']).shape)) for p in all_p)
for (is_, os_), c in shape_counts.most_common():
    print(f"  in={is_} out={os_}: {c} pairs")
