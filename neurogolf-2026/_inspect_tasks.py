"""Inspect specific UNKNOWN tasks to identify their transformation patterns."""
import json, sys
import numpy as np
from pathlib import Path

TASK_DIR = Path(__file__).parent

def load(n):
    with open(TASK_DIR / f'task{n:03d}.json') as f: return json.load(f)

def show_pair(p, label=""):
    ig, og = np.array(p['input']), np.array(p['output'])
    print(f"  --- {label} in {ig.shape} out {og.shape} ---")
    print(f"  IN:")
    for row in ig:
        print("   ", " ".join(f"{v}" for v in row))
    print(f"  OUT:")
    for row in og:
        print("   ", " ".join(f"{v}" for v in row))

tasks = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [2, 4, 5, 7, 8, 9, 10, 11, 12, 13]

for n in tasks:
    t = load(n)
    pairs = t.get('train', [])
    print(f"\n========== Task {n} ({len(pairs)} train pairs) ==========")
    for i, p in enumerate(pairs[:2]):
        show_pair(p, f"pair {i}")
