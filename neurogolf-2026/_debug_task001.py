"""Run v8 solver only on task001 to understand why fractal isn't solving."""
import json, sys, time
from pathlib import Path

NB = Path(__file__).parent / 'neurogolf-v8.ipynb'
with open(NB) as f: nb = json.load(f)
code_cells = [''.join(c['source']) if isinstance(c['source'], list) else c['source']
              for c in nb['cells'] if c['cell_type'] == 'code']
code_cells[0] = ''

# Patch task list to just [1] and add verbose print in solve_task
for i, s in enumerate(code_cells):
    if "task_files = sorted(TASK_DIR.glob" in s:
        code_cells[i] = s.replace(
            "task_files = sorted(TASK_DIR.glob('task*.json'))",
            "task_files = [TASK_DIR / 'task001.json']")
    if "zip_path = OUTPUT_DIR / 'submission.zip'" in s:
        code_cells[i] = "# skipped\n"

ns = {'__name__':'__main__'}
exec(compile('\n'.join(code_cells[:-2]), '<dbg>', 'exec'), ns)

# After all defs loaded, invoke manually on task001 with debugging
print("\n=== Manual debug: task001 ===")
with open(Path(__file__).parent / 'task001.json') as f:
    t = json.load(f)
all_p = (t.get('train',[]) + t.get('test',[]) + t.get('arc-gen',[]))[:30]
tr_p = t.get('train',[]) + t.get('test',[])

# Test detector
fst = ns['detect_fractal_self_tile'](tr_p)
print(f"detect_fractal_self_tile(tr_p)  = {fst}")

# Build model
try:
    m = ns['model_fractal_self_tile_cond'](3, 3)
    print(f"model built: {len(m.graph.node)} nodes")
except Exception as e:
    print(f"model build FAILED: {e}")
    sys.exit(1)

# Save and validate
SCRATCH = Path('/tmp/frac_scratch.onnx')
try:
    ns['save_onnx'](m, SCRATCH)
    print(f"saved: {SCRATCH.stat().st_size} bytes")
except Exception as e:
    print(f"save FAILED: {e}"); sys.exit(1)

ok, reason = ns['validate_onnx'](SCRATCH)
print(f"validate: ok={ok}  reason={reason}")

# Run on pairs
try:
    passed, total = ns['run_onnx'](SCRATCH, all_p)
    print(f"run_onnx: {passed}/{total} pass")
except Exception as e:
    print(f"run FAILED: {e}")

# Now check check_all
try:
    ok2 = ns['check_all'](SCRATCH, all_p)
    print(f"check_all: {ok2}")
except Exception as e:
    print(f"check_all FAILED: {e}")
