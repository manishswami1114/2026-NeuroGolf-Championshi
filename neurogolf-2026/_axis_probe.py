"""Measure score of axis-Gather flip models vs flat-Gather."""
import json, math, sys
from pathlib import Path
NB = Path(__file__).parent / 'neurogolf-v9.ipynb'
with open(NB) as f: nb = json.load(f)
code_cells = [''.join(c['source']) if isinstance(c['source'], list) else c['source']
              for c in nb['cells'] if c['cell_type'] == 'code']
code_cells[0] = ''
ns = {'__name__':'__main__'}
exec(compile('\n'.join(code_cells[:4]), '<p>', 'exec'), ns)

import onnx_tool
import onnx.helper as oh
import onnx.numpy_helper as onh
import numpy as np
C, H, W = 10, 30, 30
make_model = ns['make_model']
save_onnx = ns['save_onnx']

def probe(name, model):
    tmp = Path('/tmp/axis_probe.onnx')
    save_onnx(model, tmp)
    m = onnx_tool.loadmodel(str(tmp), {'verbose': False})
    g = m.graph
    g.graph_reorder_nodes(); g.shape_infer(None); g.profile()
    macs = int(sum(g.macs)); mem = int(g.memory); pars = int(g.params)
    pts = max(1.0, 25.0 - math.log(macs + mem + pars))
    print(f"{name:35s} MACs={macs:6d} mem={mem:7d} params={pars:5d} tot={macs+mem+pars:7d} pts={pts:5.2f}")

# Variant A: single-op hflip via axis=3 Gather
def hflip_axis():
    idx = onh.from_array(np.array(list(range(W-1, -1, -1)), np.int64), name='fw')
    return make_model([
        oh.make_node('Constant', [], ['fw'], value=idx),
        oh.make_node('Gather', ['input','fw'], ['output'], axis=3),
    ])

# Variant B: single-op vflip via axis=2 Gather
def vflip_axis():
    idx = onh.from_array(np.array(list(range(H-1, -1, -1)), np.int64), name='fh')
    return make_model([
        oh.make_node('Constant', [], ['fh'], value=idx),
        oh.make_node('Gather', ['input','fh'], ['output'], axis=2),
    ])

# Variant C: transpose via Transpose op (no Gather)
def transpose_op():
    return make_model([oh.make_node('Transpose', ['input'], ['output'], perm=[0,1,3,2])])

# Variant D: rot180 = two axis gathers
def rot180_axis():
    idw = onh.from_array(np.array(list(range(W-1,-1,-1)), np.int64), name='fw')
    idh = onh.from_array(np.array(list(range(H-1,-1,-1)), np.int64), name='fh')
    return make_model([
        oh.make_node('Constant', [], ['fw'], value=idw),
        oh.make_node('Gather', ['input','fw'], ['hf'], axis=3),
        oh.make_node('Constant', [], ['fh'], value=idh),
        oh.make_node('Gather', ['hf','fh'], ['output'], axis=2),
    ])

probe('hflip (axis-Gather, 1 op)', hflip_axis())
probe('vflip (axis-Gather, 1 op)', vflip_axis())
probe('transpose (Transpose op)',  transpose_op())
probe('rot180 (two axis-Gathers)', rot180_axis())
probe('hflip OLD (flat-Gather)',   ns['model_hflip']())
probe('transpose OLD (flat-Gather)', ns['model_transpose']())
probe('identity',                  ns['model_identity']())
