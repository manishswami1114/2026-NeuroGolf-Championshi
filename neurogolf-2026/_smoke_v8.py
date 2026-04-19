"""v8 smoke runner."""
import json, os, sys, time
from pathlib import Path

NB = Path(__file__).parent / 'neurogolf-v8.ipynb'
# Sample including a mix: MLP-hits, new detector candidates, fractal, scale, etc.
SAMPLE = [1, 6, 16, 32, 38, 48, 78, 87, 100, 144, 150, 164, 172, 223, 267,
          304, 307, 311, 380, 391]

with open(NB) as f: nb = json.load(f)
code_cells = [''.join(c['source']) if isinstance(c['source'], list) else c['source']
              for c in nb['cells'] if c['cell_type'] == 'code']
code_cells[0] = ''  # drop pip

for i, s in enumerate(code_cells):
    if "task_files = sorted(TASK_DIR.glob" in s:
        code_cells[i] = s.replace(
            "task_files = sorted(TASK_DIR.glob('task*.json'))",
            "task_files = [TASK_DIR / f'task{n:03d}.json' for n in SAMPLE_NUMS]")
    if "zip_path = OUTPUT_DIR / 'submission.zip'" in s:
        code_cells[i] = "# skipped\n"

glue = "\nSAMPLE_NUMS = " + repr(SAMPLE) + "\n"
full = glue + "\n".join(code_cells)
t0 = time.time()
try:
    exec(compile(full, '<smoke_v8>', 'exec'), {'__name__':'__main__'})
except SystemExit: pass
print(f"\nEXIT=0  wall={time.time()-t0:.0f}s")
