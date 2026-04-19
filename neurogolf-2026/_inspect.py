"""Quick task inspector — print first 3 input/output pairs for a list of tasks."""
import json, sys, numpy as np
from pathlib import Path
TASK_DIR = Path(__file__).parent

def show(n):
    with open(TASK_DIR / f'task{n:03d}.json') as f: t = json.load(f)
    pairs = t.get('train', [])
    print(f"\n{'='*60}\nTask {n:03d}  ({len(pairs)} train pairs)")
    for i, p in enumerate(pairs[:2]):
        ig, og = np.array(p['input']), np.array(p['output'])
        print(f"  pair {i}: in {ig.shape} -> out {og.shape}")
        print(f"   in: colors={sorted(set(ig.flatten().tolist()))}")
        print(f"  out: colors={sorted(set(og.flatten().tolist()))}")
        # Quick hints
        if ig.shape == og.shape:
            diff = (ig != og).sum()
            print(f"  diff_pixels={diff}/{ig.size} ({100*diff/ig.size:.0f}%)")
        if ig.shape[0]*2 == og.shape[0] and ig.shape[1] == og.shape[1]:
            print(f"  hint: height 2x")
        if ig.shape[0] == og.shape[0] and ig.shape[1]*2 == og.shape[1]:
            print(f"  hint: width 2x")
        if og.shape[0] == ig.shape[0]*ig.shape[0] and og.shape[1] == ig.shape[1]*ig.shape[1]:
            print(f"  hint: fractal h*h,w*w")

# First 30 UNKNOWN tasks for rapid categorization
with open(TASK_DIR / '_unknown_tasks.txt') as f:
    tasks = [int(x) for x in f.read().split()]

sample = tasks[:15]  # easier to parse than all 373 at once
for n in sample:
    show(n)
