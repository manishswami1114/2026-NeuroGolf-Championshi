"""Measure actual macs/memory/params for each v9 model type to find what's scoring low."""
import json, sys, math
from pathlib import Path
NB = Path(__file__).parent / 'neurogolf-v9.ipynb'
with open(NB) as f: nb = json.load(f)
code_cells = [''.join(c['source']) if isinstance(c['source'], list) else c['source']
              for c in nb['cells'] if c['cell_type'] == 'code']
code_cells[0] = ''  # drop pip

ns = {'__name__':'__main__'}
# Only load up through handcrafted models + helpers (not solve_task)
exec(compile('\n'.join(code_cells[:4]), '<p>', 'exec'), ns)

import onnx_tool
def score(model):
    tmp = Path('/tmp/probe.onnx')
    ns['save_onnx'](model, tmp)
    m = onnx_tool.loadmodel(str(tmp), {'verbose': False})
    g = m.graph
    g.graph_reorder_nodes(); g.shape_infer(None); g.profile()
    macs, mem, pars = int(sum(g.macs)), int(g.memory), int(g.params)
    pts = max(1.0, 25.0 - math.log(macs + mem + pars))
    return macs, mem, pars, pts

models = [
    ('identity', lambda: ns['model_identity']()),
    ('color_remap {0->1}', lambda: ns['model_color_remap']({0:1, 1:0})),
    ('mirror_h 5x5', lambda: ns['model_mirror_h'](5, 5)),
    ('mirror_quad 5x5', lambda: ns['model_mirror_quad'](5, 5)),
    ('double_h 5x5', lambda: ns['model_double_h'](5, 5)),
    ('scale_up k=2 5x5', lambda: ns['model_scale_up'](2, 5, 5)),
    ('scale_up k=3 3x3', lambda: ns['model_scale_up'](3, 3, 3)),
    ('scale_down k=2 10x10', lambda: ns['model_scale_down'](2, 10, 10)),
    ('recolor_all c=3', lambda: ns['model_recolor_all'](3)),
    ('keep_only c=3', lambda: ns['model_keep_only_color'](3)),
    ('fill_bg c=3', lambda: ns['model_fill_bg'](3)),
    ('const_1x1 c=3', lambda: ns['model_const_1x1'](3)),
    ('fractal 3x3', lambda: ns['model_fractal_self_tile_cond'](3, 3)),
]

print(f"{'model':25s} {'MACs':>8s} {'mem':>8s} {'params':>8s} {'total':>8s} {'pts':>6s}")
print("-"*70)
for name, mk in models:
    try:
        macs, mem, pars, pts = score(mk())
        print(f"{name:25s} {macs:8d} {mem:8d} {pars:8d} {macs+mem+pars:8d} {pts:6.2f}")
    except Exception as e:
        print(f"{name:25s} ERROR {e}")
