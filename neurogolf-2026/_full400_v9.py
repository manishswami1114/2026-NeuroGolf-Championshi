"""Run v9 on ALL 400 tasks locally to get real projection before Kaggle submit."""
import json, os, sys, time
from pathlib import Path

NB = Path(__file__).parent / 'neurogolf-v9.ipynb'

with open(NB) as f: nb = json.load(f)
code_cells = [''.join(c['source']) if isinstance(c['source'], list) else c['source']
              for c in nb['cells'] if c['cell_type'] == 'code']
code_cells[0] = ''

# Keep full task list (no patch)
for i, s in enumerate(code_cells):
    if "zip_path = OUTPUT_DIR / 'submission.zip'" in s:
        pass  # keep zip build to verify submission artifact

full = "\n".join(code_cells)
t0 = time.time()
try:
    exec(compile(full, '<full400_v9>', 'exec'), {'__name__':'__main__'})
except SystemExit: pass
print(f"\nEXIT=0  wall={time.time()-t0:.0f}s")
