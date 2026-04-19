"""v9 smoke runner."""
import json, os, sys, time
from pathlib import Path

NB = Path(__file__).parent / 'neurogolf-v9.ipynb'
# Include same 20 as v8 + tasks matching v9 new detectors:
#   - 2 recolor_all candidates (58, 171)
#   - 3 mirror_quad (83, 142, 152)
#   - 1 count_color_1x1 maybe (48)
#   - 1 crop_bbox (31)
#   - 1 gravity_down (32)
SAMPLE = [1, 6, 16, 32, 38, 48, 58, 83, 87, 100, 142, 150, 152, 164, 171, 172,
          210, 223, 267, 307, 311, 380]

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
    exec(compile(full, '<smoke_v9>', 'exec'), {'__name__':'__main__'})
except SystemExit: pass
print(f"\nEXIT=0  wall={time.time()-t0:.0f}s")
