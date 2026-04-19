"""Ad-hoc smoke runner for v7: extracts cells from neurogolf-v7.ipynb,
overrides the task list to a small sample that includes known MLP tasks
(001, 006, 038, 048, 100, 144, 391 solved by v6), and runs end-to-end.

Writes per-task lines matching the format of earlier smoke logs so they
can be compared directly.
"""
import json, os, sys, time, re
from pathlib import Path

NB = Path(__file__).parent / 'neurogolf-v7.ipynb'
SAMPLE = [1, 6, 16, 32, 38, 48, 100, 144, 391, 150, 170, 300, 130, 200, 380]

with open(NB) as f:
    nb = json.load(f)

# Concatenate all code cells except the "Run on all 400 tasks" driver (we
# replace the task list).  We also skip the final zip cell.
code_cells = [''.join(c['source']) if isinstance(c['source'], list) else c['source']
              for c in nb['cells'] if c['cell_type'] == 'code']

# Drop the !pip line (opaque to plain python).
code_cells[0] = ''

# The driver cell contains 'task_files = sorted(TASK_DIR.glob'.
# Replace its glob line with a filter using SAMPLE.
for i, s in enumerate(code_cells):
    if "task_files = sorted(TASK_DIR.glob" in s:
        patched = s.replace(
            "task_files = sorted(TASK_DIR.glob('task*.json'))",
            "task_files = [TASK_DIR / f'task{n:03d}.json' for n in SAMPLE_NUMS]"
        )
        code_cells[i] = patched
    # Nuke the final zip cell (we don't need a zip for smoke).
    if "zip_path = OUTPUT_DIR / 'submission.zip'" in s:
        code_cells[i] = "# skipped in smoke\n"

glue = "\nSAMPLE_NUMS = " + repr(SAMPLE) + "\n"
full = glue + "\n".join(code_cells)

# Run.
t0 = time.time()
ns = {'__name__': '__main__'}
try:
    exec(compile(full, '<smoke_v7>', 'exec'), ns)
except SystemExit:
    pass
print(f"\nEXIT=0  wall={time.time()-t0:.0f}s")
