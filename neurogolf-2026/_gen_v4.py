"""Generate neurogolf-v4.ipynb.

Critical fixes over v3:
- Opset 10 / IR 10 (v3 used opset 17 / IR 8 -- incompatible with Kaggle scorer)
- Pad uses attribute `pads` (opset-10 Pad-2), not input
- PyTorch export with opset_version=10, post-processed to ir_version=10
- onnx.checker.check_model() + op_type whitelist before accepting a model
- Verbose score_model -- no silent 1.0 returns
- !pip install onnx-tool so score works on Kaggle

Features added over v3:
- model_stack_v / stack_h (output = input tiled N times)
- model_mirror_h / mirror_v (output = input + flip(input) concat)
- model_single_color (output = constant solid color)
- model_color_filter (keep only certain colors)
- model_half (top/bottom/left/right half of input)
- detectors for each of the above
"""
import json, os

CELLS = []
def md(s):   CELLS.append({"cell_type":"markdown","id":f"c{len(CELLS)}","metadata":{},"source":s})
def code(s): CELLS.append({"cell_type":"code","id":f"c{len(CELLS)}","metadata":{},
                           "outputs":[],"execution_count":None,"source":s})

md("""# NeuroGolf 2026 -- Competitive Solver v4

**Critical bugfixes over v3 (which errored on Kaggle submission):**
- **Opset 10 / IR 10** -- matches `neurogolf_utils._IR_VERSION = 10`. v3's opset 17 was rejected by the Kaggle scorer's onnx_tool.
- **Pad as attribute** (opset-10 semantics), not input.
- **PyTorch export at opset 10**, post-processed to ir_version=10.
- **Runtime validation** before saving: `onnx.checker.check_model()` + op-type whitelist.
- **Verbose `score_model`** -- prints actual errors instead of silently returning 1.0.
- **`!pip install onnx-tool`** so scoring runs on Kaggle.

**New detectors (targeting 138 different-size tasks):**
- `stack_v` / `stack_h`: output is input repeated N× vertically/horizontally
- `mirror_h` / `mirror_v`: output = [input | fliplr(input)] or [input ; flipud(input)]
- `single_color`: output is solid color
- `color_filter`: output keeps one/few colors, zeros the rest
- `half`: output is top/bottom/left/right half of input
""")

code("""!pip install onnx-tool -q 2>/dev/null || true
""")

code("""import os, json, math, time, zipfile, warnings, inspect as _inspect
import numpy as np
from pathlib import Path
import onnx, onnxruntime
import onnx.helper as oh
import onnx.numpy_helper as onh
from onnx import TensorProto
import torch, torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings('ignore')

# Kaggle vs local paths
_KAGGLE = Path('/kaggle/input/competitions/neurogolf-2026')
if _KAGGLE.exists():
    TASK_DIR   = _KAGGLE
    OUTPUT_DIR = Path('/kaggle/working')
else:
    TASK_DIR   = Path('./')
    OUTPUT_DIR = Path('./submission_v4')
SUB_DIR = OUTPUT_DIR / 'submission'
SUB_DIR.mkdir(parents=True, exist_ok=True)

C, H, W     = 10, 30, 30
OPSET       = oh.make_opsetid('', 10)   # MATCHES neurogolf_utils._OPSET_IMPORTS
IR_VER      = 10                         # MATCHES neurogolf_utils._IR_VERSION
MAX_BYTES   = int(1.44 * 1024 * 1024)
CNN_TIMEOUT = 30.0
_BANNED_OPS = {'LOOP','SCAN','NONZERO','UNIQUE','SCRIPT','FUNCTION'}

DEVICE = (torch.device('cuda') if torch.cuda.is_available() else
          torch.device('mps')  if torch.backends.mps.is_available() else
          torch.device('cpu'))
print(f'Device: {DEVICE}  |  Opset: 10  |  IR: 10  |  Task dir: {TASK_DIR}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    torch.backends.cudnn.benchmark = True
""")

code("""def load_task(path):
    with open(path) as f: return json.load(f)

def g2t(grid):
    g = np.array(grid, dtype=np.int32)
    h, w = g.shape
    t = np.zeros((C, H, W), dtype=np.float32)
    for r in range(h):
        for c in range(w):
            v = g[r,c]
            if 0 <= v <= 9: t[v, r, c] = 1.0
    return t

def g2t_batch(pairs, key):
    return np.stack([g2t(p[key]) for p in pairs])

def t2grid_strict(t_thresholded):
    '''Match neurogolf_utils.convert_from_numpy EXACTLY:
    - cell with 0 channels > 0 -> 10 (no color)
    - cell with 1 channel > 0 -> that color (0..9)
    - cell with 2+ channels > 0 -> 11 (too many)
    - trim trailing 10-cells from rows, then trim trailing empty rows.'''
    arr = t_thresholded[0] if t_thresholded.ndim == 4 else t_thresholded
    _, height, width = arr.shape
    grid = []
    for r in range(height):
        cells = []
        for c in range(width):
            colors = [ch for ch in range(C) if arr[ch, r, c] == 1]
            if len(colors) == 1:   cells.append(colors[0])
            elif len(colors) == 0: cells.append(10)
            else:                  cells.append(11)
        while cells and cells[-1] == 10:
            cells.pop()
        grid.append(cells)
    while grid and not grid[-1]:
        grid.pop()
    return grid

def make_model(nodes, inits=None):
    X = oh.make_tensor_value_info('input',  TensorProto.FLOAT, [1,C,H,W])
    Y = oh.make_tensor_value_info('output', TensorProto.FLOAT, [1,C,H,W])
    graph = oh.make_graph(nodes, 'g', [X], [Y], initializer=inits or [])
    model = oh.make_model(graph, opset_imports=[OPSET])
    model.ir_version = IR_VER
    return model

def validate_onnx(model_or_path):
    '''Run onnx.checker + banned-op check. Returns (ok, reason).'''
    try:
        m = onnx.load(str(model_or_path)) if isinstance(model_or_path, (str, Path)) else model_or_path
        onnx.checker.check_model(m)
        for node in m.graph.node:
            if node.op_type.upper() in _BANNED_OPS:
                return False, f'banned_op:{node.op_type}'
        return True, 'ok'
    except Exception as e:
        return False, f'checker:{type(e).__name__}:{str(e)[:80]}'

def save_onnx(model, path):
    with open(path, 'wb') as f: f.write(model.SerializeToString())
    return Path(path).stat().st_size

def run_onnx(model_or_path, pairs):
    try:
        if isinstance(model_or_path, (str, Path)):
            sess = onnxruntime.InferenceSession(str(model_or_path), providers=['CPUExecutionProvider'])
        else:
            sess = onnxruntime.InferenceSession(model_or_path.SerializeToString(), providers=['CPUExecutionProvider'])
    except Exception: return 0, len(pairs)
    correct = 0
    for p in pairs:
        inp  = g2t(p['input'])[np.newaxis]
        og   = np.array(p['output'])
        oh_, ow_ = og.shape
        try:
            raw  = sess.run(None, {'input': inp})[0]
            pred = (raw > 0.0).astype(np.float32)   # MATCHES run_network threshold
            if t2grid_strict(pred) == p['output']: correct += 1
        except Exception: pass
    return correct, len(pairs)

def check_all(model, all_pairs):
    c, t = run_onnx(model, all_pairs)
    return c == t

def score_model(path, verbose=False):
    try:
        import onnx_tool
        m = onnx_tool.loadmodel(str(path), {'verbose': False})
        g = m.graph
        g.graph_reorder_nodes(); g.shape_infer(None); g.profile()
        cost = int(sum(g.macs)) + int(g.memory) + int(g.params)
        return max(1.0, 25.0 - math.log(max(cost, 1)))
    except Exception as e:
        if verbose:
            print(f'  [score_model fallback: {type(e).__name__}: {e}]')
        # Manual fallback: estimate params+memory from protobuf
        try:
            m = onnx.load(str(path))
            n_params = 0
            n_mem = 0
            for init in m.graph.initializer:
                arr = onh.to_array(init)
                n_params += arr.size
                n_mem    += arr.nbytes
            # rough MAC estimate: 0 (fallback)
            cost = n_mem + n_params
            return max(1.0, 25.0 - math.log(max(cost, 1)))
        except Exception:
            return 1.0

def _onnx_export(model, path):
    '''Export PyTorch model at opset 10, then rewrite ir_version to 10.'''
    model.cpu().eval()
    dummy = torch.randn(1, C, H, W)
    kw = dict(opset_version=10, input_names=['input'], output_names=['output'],
              do_constant_folding=True)
    sig = _inspect.signature(torch.onnx.export).parameters
    if 'dynamo' in sig: kw['dynamo'] = False
    torch.onnx.export(model, dummy, str(path), **kw)
    # Normalize ir_version for Kaggle scorer
    m = onnx.load(str(path))
    m.ir_version = IR_VER
    # Ensure opset import is exactly ('', 10)
    del m.opset_import[:]
    m.opset_import.append(oh.make_opsetid('', 10))
    with open(path, 'wb') as f: f.write(m.SerializeToString())

print('Helpers ready.')
""")

code("""# --- Handcrafted ONNX model builders (all opset-10 compliant) ---

def _gather_model(indices):
    sh_chw = onh.from_array(np.array([1,C,H*W], np.int64), name='sh_chw')
    sh_out = onh.from_array(np.array([1,C,H,W], np.int64), name='sh_out')
    gi     = onh.from_array(indices.astype(np.int64),       name='gi')
    return make_model([
        oh.make_node('Constant', [], ['sh_chw'], value=sh_chw),
        oh.make_node('Reshape',  ['input','sh_chw'], ['flat']),
        oh.make_node('Constant', [], ['gi'],     value=gi),
        oh.make_node('Gather',   ['flat','gi'],  ['gathered'], axis=2),
        oh.make_node('Constant', [], ['sh_out'], value=sh_out),
        oh.make_node('Reshape',  ['gathered','sh_out'], ['output']),
    ])

def _perm_from_fn(fn):
    idx = np.zeros(H*W, np.int64)
    for r in range(H):
        for c in range(W):
            nr, nc = fn(r, c)
            if 0 <= nr < H and 0 <= nc < W:
                idx[r*W+c] = nr*W+nc
    return idx

def model_identity():
    return make_model([oh.make_node('Identity', ['input'], ['output'])])

def model_color_remap(cmap):
    Wt = np.zeros((C,C,1,1), np.float32)
    for s, d in cmap.items(): Wt[d,s,0,0] = 1.0
    for ch in range(C):
        if ch not in cmap: Wt[ch,ch,0,0] = 1.0
    wt = onh.from_array(Wt, name='W')
    return make_model([
        oh.make_node('Constant', [], ['W'], value=wt),
        oh.make_node('Conv', ['input','W'], ['output'], kernel_shape=[1,1], pads=[0,0,0,0]),
    ])

model_rot90   = lambda: _gather_model(_perm_from_fn(lambda r,c:(c,H-1-r)))
model_rot180  = lambda: _gather_model(_perm_from_fn(lambda r,c:(H-1-r,W-1-c)))
model_rot270  = lambda: _gather_model(_perm_from_fn(lambda r,c:(W-1-c,r)))
model_hflip   = lambda: _gather_model(_perm_from_fn(lambda r,c:(r,W-1-c)))
model_vflip   = lambda: _gather_model(_perm_from_fn(lambda r,c:(H-1-r,c)))
model_hvflip  = lambda: _gather_model(_perm_from_fn(lambda r,c:(H-1-r,W-1-c)))
model_transpose    = lambda: _gather_model(_perm_from_fn(lambda r,c:(c,r)))
model_antitranspose= lambda: _gather_model(_perm_from_fn(lambda r,c:(W-1-c,H-1-r)))

def model_scale(factor):
    idx = np.zeros(H*W, np.int64)
    for r in range(H):
        for c in range(W):
            idx[r*W+c] = (r//factor)*W + (c//factor)
    return _gather_model(idx)

def model_tile(tr, tc, ih, iw):
    th, tw = ih//tr, iw//tc
    idx = np.zeros(H*W, np.int64)
    for r in range(H):
        for c in range(W):
            idx[r*W+c] = (r % th)*W + (c % tw)
    return _gather_model(idx)

def model_crop(r0, c0, oh_, ow_):
    '''Crop using gather + mask (output in top-left at (0..oh_, 0..ow_); zero elsewhere).'''
    idx = np.zeros(H*W, np.int64)
    for r in range(H):
        for c in range(W):
            if r < oh_ and c < ow_:
                sr, sc = r0+r, c0+c
                idx[r*W+c] = sr*W+sc if sr<H and sc<W else 0
    mask = np.zeros((1,C,H,W), np.float32)
    mask[0,:,:oh_,:ow_] = 1.0
    mk   = onh.from_array(mask, name='mask')
    sh_c = onh.from_array(np.array([1,C,H*W],np.int64), name='sh_c')
    sh_o = onh.from_array(np.array([1,C,H,W],np.int64), name='sh_o')
    gi   = onh.from_array(idx, name='gi')
    return make_model([
        oh.make_node('Constant',[],['sh_c'], value=sh_c),
        oh.make_node('Reshape', ['input','sh_c'],['flat']),
        oh.make_node('Constant',[],['gi'],   value=gi),
        oh.make_node('Gather',  ['flat','gi'],['gath'],axis=2),
        oh.make_node('Constant',[],['sh_o'], value=sh_o),
        oh.make_node('Reshape', ['gath','sh_o'],['raw']),
        oh.make_node('Constant',[],['mask'], value=mk),
        oh.make_node('Mul',     ['raw','mask'],['output']),
    ])

def model_const(output_tensor):
    ct = onh.from_array(output_tensor.astype(np.float32), name='ct')
    zr = onh.from_array(np.zeros((1,C,H,W), np.float32),  name='zr')
    return make_model([
        oh.make_node('Constant',[],['ct'],value=ct),
        oh.make_node('Constant',[],['zr'],value=zr),
        oh.make_node('Mul', ['input','zr'],  ['zeroed']),
        oh.make_node('Add', ['zeroed','ct'], ['output']),
    ])

def model_roll(dr, dc, ih, iw):
    idx = np.zeros(H*W, np.int64)
    for r in range(H):
        for c in range(W):
            if r < ih and c < iw:
                idx[r*W+c] = ((r-dr) % ih)*W + ((c-dc) % iw)
    return _gather_model(idx)

def model_remap_geom(cmap, geom_idx):
    Wt   = np.zeros((C,C,1,1), np.float32)
    for s, d in cmap.items(): Wt[d,s,0,0] = 1.0
    for ch in range(C):
        if ch not in cmap: Wt[ch,ch,0,0] = 1.0
    wt   = onh.from_array(Wt, name='W2')
    sh_c = onh.from_array(np.array([1,C,H*W],np.int64), name='sh_c')
    sh_o = onh.from_array(np.array([1,C,H,W],np.int64), name='sh_o')
    gi   = onh.from_array(geom_idx, name='gi2')
    return make_model([
        oh.make_node('Constant',[],['W2'],  value=wt),
        oh.make_node('Conv',['input','W2'],['remapped'],kernel_shape=[1,1],pads=[0,0,0,0]),
        oh.make_node('Constant',[],['sh_c'],value=sh_c),
        oh.make_node('Reshape',['remapped','sh_c'],['flat2']),
        oh.make_node('Constant',[],['gi2'], value=gi),
        oh.make_node('Gather',  ['flat2','gi2'],['gath2'],axis=2),
        oh.make_node('Constant',[],['sh_o'],value=sh_o),
        oh.make_node('Reshape', ['gath2','sh_o'],['output']),
    ])

# -- NEW v4 MODELS -----------------------------------------------------

def model_stack_v(ih, iw, n):
    '''Output = vertical stack of n copies of input. Result size (n*ih, iw).'''
    oh_, ow_ = n*ih, iw
    idx = np.zeros(H*W, np.int64)
    for r in range(H):
        for c in range(W):
            if r < oh_ and c < ow_:
                sr = r % ih
                idx[r*W+c] = sr*W + c
    mask = np.zeros((1,C,H,W), np.float32)
    mask[0,:,:oh_,:ow_] = 1.0
    mk   = onh.from_array(mask, name='mask')
    sh_c = onh.from_array(np.array([1,C,H*W],np.int64), name='sh_c')
    sh_o = onh.from_array(np.array([1,C,H,W],np.int64), name='sh_o')
    gi   = onh.from_array(idx, name='gi')
    return make_model([
        oh.make_node('Constant',[],['sh_c'], value=sh_c),
        oh.make_node('Reshape', ['input','sh_c'],['flat']),
        oh.make_node('Constant',[],['gi'],   value=gi),
        oh.make_node('Gather',  ['flat','gi'],['gath'],axis=2),
        oh.make_node('Constant',[],['sh_o'], value=sh_o),
        oh.make_node('Reshape', ['gath','sh_o'],['raw']),
        oh.make_node('Constant',[],['mask'], value=mk),
        oh.make_node('Mul',     ['raw','mask'],['output']),
    ])

def model_stack_h(ih, iw, n):
    oh_, ow_ = ih, n*iw
    idx = np.zeros(H*W, np.int64)
    for r in range(H):
        for c in range(W):
            if r < oh_ and c < ow_:
                sc = c % iw
                idx[r*W+c] = r*W + sc
    mask = np.zeros((1,C,H,W), np.float32)
    mask[0,:,:oh_,:ow_] = 1.0
    mk   = onh.from_array(mask, name='mask')
    sh_c = onh.from_array(np.array([1,C,H*W],np.int64), name='sh_c')
    sh_o = onh.from_array(np.array([1,C,H,W],np.int64), name='sh_o')
    gi   = onh.from_array(idx, name='gi')
    return make_model([
        oh.make_node('Constant',[],['sh_c'], value=sh_c),
        oh.make_node('Reshape', ['input','sh_c'],['flat']),
        oh.make_node('Constant',[],['gi'],   value=gi),
        oh.make_node('Gather',  ['flat','gi'],['gath'],axis=2),
        oh.make_node('Constant',[],['sh_o'], value=sh_o),
        oh.make_node('Reshape', ['gath','sh_o'],['raw']),
        oh.make_node('Constant',[],['mask'], value=mk),
        oh.make_node('Mul',     ['raw','mask'],['output']),
    ])

def model_mirror_h(ih, iw):
    '''Output = [input | fliplr(input)]. Size (ih, 2*iw).'''
    oh_, ow_ = ih, 2*iw
    idx = np.zeros(H*W, np.int64)
    for r in range(H):
        for c in range(W):
            if r < oh_ and c < ow_:
                if c < iw: sc = c
                else:      sc = iw - 1 - (c - iw)
                idx[r*W+c] = r*W + sc
    mask = np.zeros((1,C,H,W), np.float32)
    mask[0,:,:oh_,:ow_] = 1.0
    mk   = onh.from_array(mask, name='mask')
    sh_c = onh.from_array(np.array([1,C,H*W],np.int64), name='sh_c')
    sh_o = onh.from_array(np.array([1,C,H,W],np.int64), name='sh_o')
    gi   = onh.from_array(idx, name='gi')
    return make_model([
        oh.make_node('Constant',[],['sh_c'], value=sh_c),
        oh.make_node('Reshape', ['input','sh_c'],['flat']),
        oh.make_node('Constant',[],['gi'],   value=gi),
        oh.make_node('Gather',  ['flat','gi'],['gath'],axis=2),
        oh.make_node('Constant',[],['sh_o'], value=sh_o),
        oh.make_node('Reshape', ['gath','sh_o'],['raw']),
        oh.make_node('Constant',[],['mask'], value=mk),
        oh.make_node('Mul',     ['raw','mask'],['output']),
    ])

def model_mirror_v(ih, iw):
    oh_, ow_ = 2*ih, iw
    idx = np.zeros(H*W, np.int64)
    for r in range(H):
        for c in range(W):
            if r < oh_ and c < ow_:
                if r < ih: sr = r
                else:      sr = ih - 1 - (r - ih)
                idx[r*W+c] = sr*W + c
    mask = np.zeros((1,C,H,W), np.float32)
    mask[0,:,:oh_,:ow_] = 1.0
    mk   = onh.from_array(mask, name='mask')
    sh_c = onh.from_array(np.array([1,C,H*W],np.int64), name='sh_c')
    sh_o = onh.from_array(np.array([1,C,H,W],np.int64), name='sh_o')
    gi   = onh.from_array(idx, name='gi')
    return make_model([
        oh.make_node('Constant',[],['sh_c'], value=sh_c),
        oh.make_node('Reshape', ['input','sh_c'],['flat']),
        oh.make_node('Constant',[],['gi'],   value=gi),
        oh.make_node('Gather',  ['flat','gi'],['gath'],axis=2),
        oh.make_node('Constant',[],['sh_o'], value=sh_o),
        oh.make_node('Reshape', ['gath','sh_o'],['raw']),
        oh.make_node('Constant',[],['mask'], value=mk),
        oh.make_node('Mul',     ['raw','mask'],['output']),
    ])

def model_single_color(color_idx, oh_, ow_):
    '''Output = solid grid of color_idx at shape (oh_, ow_), zeros elsewhere.'''
    out = np.zeros((1,C,H,W), np.float32)
    out[0, color_idx, :oh_, :ow_] = 1.0
    ct = onh.from_array(out, name='ct')
    zr = onh.from_array(np.zeros((1,C,H,W), np.float32), name='zr')
    return make_model([
        oh.make_node('Constant',[],['ct'],value=ct),
        oh.make_node('Constant',[],['zr'],value=zr),
        oh.make_node('Mul', ['input','zr'],  ['zeroed']),
        oh.make_node('Add', ['zeroed','ct'], ['output']),
    ])

def model_color_filter(keep_colors):
    '''Zero out all channels NOT in keep_colors; preserve others unchanged (1x1 conv).'''
    Wt = np.zeros((C,C,1,1), np.float32)
    for ch in keep_colors:
        Wt[ch,ch,0,0] = 1.0
    wt = onh.from_array(Wt, name='W')
    return make_model([
        oh.make_node('Constant', [], ['W'], value=wt),
        oh.make_node('Conv', ['input','W'], ['output'], kernel_shape=[1,1], pads=[0,0,0,0]),
    ])

print('Handcrafted models ready (opset 10).')
""")

code("""# --- Linear solver using Pad-2 (opset 10 attribute form) ---

def _model_linear_small(W_np, ih, iw, oh_, ow_):
    '''Replace Slice with Gather (onnx_tool-compatible shape_infer).
    Flow: Reshape -> Gather (pick ih*iw positions) -> Reshape -> Gemm -> Reshape -> Pad.'''
    in_spatial  = ih * iw
    out_spatial = oh_ * ow_
    in_dim      = C * in_spatial
    out_dim     = C * out_spatial
    # Indices to pull (0..ih) x (0..iw) from H*W flat layout
    slice_idx = np.array([r*W + c for r in range(ih) for c in range(iw)], np.int64)
    sh_chw = onh.from_array(np.array([1, C, H*W],        np.int64), name='sh_chw')
    gi     = onh.from_array(slice_idx,                              name='slice_gi')
    sh_fl  = onh.from_array(np.array([1, in_dim],        np.int64), name='sh_fl')
    wt     = onh.from_array(W_np.astype(np.float32),                name='W')
    bias   = onh.from_array(np.zeros(out_dim, np.float32),          name='B')
    sh_sm  = onh.from_array(np.array([1, C, oh_, ow_],   np.int64), name='sh_sm')
    pads_attr = [0,0,0,0, 0,0, H-oh_, W-ow_]  # opset-10 Pad uses attribute
    return make_model([
        oh.make_node('Constant',[],['sh_chw'],value=sh_chw),
        oh.make_node('Reshape',['input','sh_chw'],['flat_hw']),
        oh.make_node('Constant',[],['slice_gi'],value=gi),
        oh.make_node('Gather',['flat_hw','slice_gi'],['sliced'], axis=2),
        oh.make_node('Constant',[],['sh_fl'],value=sh_fl),
        oh.make_node('Reshape',['sliced','sh_fl'],['flat']),
        oh.make_node('Constant',[],['W'],value=wt),
        oh.make_node('Constant',[],['B'],value=bias),
        oh.make_node('Gemm',['flat','W','B'],['out_sm'],transB=0,alpha=1.0,beta=0.0),
        oh.make_node('Constant',[],['sh_sm'],value=sh_sm),
        oh.make_node('Reshape',['out_sm','sh_sm'],['small']),
        oh.make_node('Pad',['small'],['output'], mode='constant', pads=pads_attr, value=0.0),
    ])

def _all_same_size(pairs):
    sizes = set()
    for p in pairs:
        sizes.add((np.array(p['input']).shape, np.array(p['output']).shape))
    return len(sizes) == 1, list(sizes)[0] if len(sizes) == 1 else None

def try_linear_solve(all_pairs):
    ok, sz = _all_same_size(all_pairs)
    if not ok: return None
    (ih,iw),(oh_,ow_) = sz
    in_dim  = C * ih * iw
    out_dim = C * oh_ * ow_
    if in_dim * out_dim * 4 > MAX_BYTES: return None
    if len(all_pairs) < 2: return None

    def g2small(grid, h, w):
        t = np.zeros((C, h, w), np.float32)
        for r in range(h):
            for c in range(w):
                t[int(grid[r][c]), r, c] = 1.0
        return t.flatten()

    X = np.stack([g2small(p['input'],  ih, iw)  for p in all_pairs])
    Y = np.stack([g2small(p['output'], oh_, ow_) for p in all_pairs])
    try:
        W, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        pred = ((X @ W) > 0.0).astype(np.float32)
        if np.max(np.abs(pred - Y)) > 0.05: return None
    except Exception: return None
    return _model_linear_small(W, ih, iw, oh_, ow_)

print('Linear solver ready (Pad-2 attribute form).')
""")

code("""# --- Trained CNN (PyTorch -> ONNX opset 10) ---

class ConvNet(nn.Module):
    def __init__(self, kind, hidden=8, k=3, blocks=2):
        super().__init__()
        p = k//2
        if kind == 'conv1':
            self.net = nn.Conv2d(C, C, k, padding=p, bias=False)
        elif kind == 'bneck':
            self.net = nn.Sequential(
                nn.Conv2d(C,hidden,1,bias=False), nn.ReLU(),
                nn.Conv2d(hidden,hidden,k,padding=p,bias=False), nn.ReLU(),
                nn.Conv2d(hidden,C,1,bias=False))
        elif kind == 'res':
            layers = [nn.Conv2d(C,hidden,k,padding=p,bias=False), nn.ReLU()]
            for _ in range(blocks-1):
                layers += [nn.Conv2d(hidden,hidden,k,padding=p,bias=False), nn.ReLU()]
            layers.append(nn.Conv2d(hidden,C,k,padding=p,bias=False))
            self.body = nn.Sequential(*layers)
            self.net  = None
        elif kind == 'dilated':
            self.net = nn.Sequential(
                nn.Conv2d(C,hidden,k,padding=p,     dilation=1,bias=False), nn.ReLU(),
                nn.Conv2d(hidden,hidden,k,padding=p*2,dilation=2,bias=False), nn.ReLU(),
                nn.Conv2d(hidden,hidden,k,padding=p*4,dilation=4,bias=False), nn.ReLU(),
                nn.Conv2d(hidden,C,1,bias=False))
        self.kind = kind

    def forward(self, x):
        if self.kind == 'res': return x + self.body(x)
        return self.net(x)

ARCH_LIST = [
    ('conv1',   dict(kind='conv1',   k=1)),
    ('conv3',   dict(kind='conv1',   k=3)),
    ('conv5',   dict(kind='conv1',   k=5)),
    ('bnk4k3',  dict(kind='bneck',   hidden=4,  k=3)),
    ('bnk8k3',  dict(kind='bneck',   hidden=8,  k=3)),
    ('bnk8k5',  dict(kind='bneck',   hidden=8,  k=5)),
    ('res8b2',  dict(kind='res',     hidden=8,  k=3, blocks=2)),
    ('dil8',    dict(kind='dilated', hidden=8,  k=3)),
]

def train_conv(train_pairs, arch_kwargs,
               max_epochs=2000, lr=0.01, patience=200, early_fail=200,
               timeout=CNN_TIMEOUT):
    t0 = time.time()
    for p in train_pairs:
        if np.array(p['input']).shape != np.array(p['output']).shape:
            return None, False
    X = torch.tensor(g2t_batch(train_pairs,'input')).to(DEVICE)
    Y = torch.tensor(g2t_batch(train_pairs,'output')).to(DEVICE)
    n = len(train_pairs)
    model = ConvNet(**arch_kwargs).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sch   = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=500, T_mult=2)
    best_correct, best_state, no_imp = 0, None, 0

    def accuracy():
        model.eval()
        with torch.no_grad():
            pred = (model(X) > 0).float()
            return int(torch.all(pred == Y, dim=(1,2,3)).sum().item())

    for epoch in range(max_epochs):
        if time.time() - t0 > timeout: break
        model.train()
        opt.zero_grad()
        out  = model(X)
        loss = F.mse_loss(out, Y) + F.binary_cross_entropy_with_logits(out * 5, Y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        if (epoch+1) % 25 == 0:
            c = accuracy()
            if (epoch+1) >= early_fail and best_correct == 0: return None, False
            if c > best_correct:
                best_correct = c
                best_state   = {k: v.cpu().clone() for k,v in model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 25
            if c == n: break
            if no_imp >= patience: break

    if best_state is None or best_correct < n: return None, False
    model.load_state_dict(best_state)
    model.cpu()
    return model, True

print('Trained conv solver ready (export opset 10).')
""")

code("""# --- Detectors ---

def detect_identity(pairs):
    ok, sz = _all_same_size(pairs)
    if not ok: return False
    return all(np.array_equal(np.array(p['input']), np.array(p['output'])) for p in pairs)

def detect_color_remap(pairs):
    ok, sz = _all_same_size(pairs)
    if not ok or sz[0] != sz[1]: return None
    cmap = {}
    for p in pairs:
        a = np.array(p['input']).flatten()
        b = np.array(p['output']).flatten()
        for s, d in zip(a, b):
            s, d = int(s), int(d)
            if s in cmap and cmap[s] != d: return None
            cmap[s] = d
    return cmap if cmap else None

def detect_rotation(pairs):
    for angle, k in [(90,1),(180,2),(270,3)]:
        if all(np.array_equal(np.rot90(np.array(p['input']),k), np.array(p['output']))
               for p in pairs):
            return angle
    return None

def detect_flip(pairs):
    for ax, name in [(1,'hflip'),(0,'vflip'),(-1,'hvflip')]:
        def flipped(ig, ax=ax):
            return np.flip(np.flip(ig,0),1) if ax==-1 else np.flip(ig,ax)
        if all(np.array_equal(flipped(np.array(p['input'])), np.array(p['output']))
               for p in pairs):
            return name
    return None

def detect_transpose(pairs):
    return all(np.array_equal(np.array(p['input']).T, np.array(p['output'])) for p in pairs)

def detect_antitranspose(pairs):
    return all(np.array_equal(np.fliplr(np.flipud(np.array(p['input']))).T,
                              np.array(p['output'])) for p in pairs)

def detect_scale(pairs):
    for f in [2,3,4,5]:
        if all(np.array_equal(
               np.repeat(np.repeat(np.array(p['input']),f,0),f,1),
               np.array(p['output'])) for p in pairs):
            return f
    return None

def detect_tile(pairs):
    p0 = pairs[0]
    ih,iw    = np.array(p0['input']).shape
    oh_,ow_  = np.array(p0['output']).shape
    if oh_%ih != 0 or ow_%iw != 0 or (oh_//ih, ow_//iw)==(1,1): return None
    tr, tc = oh_//ih, ow_//iw
    if all(np.array_equal(np.tile(np.array(p['input']),(tr,tc)),
                          np.array(p['output'])) for p in pairs):
        return tr, tc, ih, iw
    return None

def detect_crop(pairs):
    p0 = pairs[0]
    ig0, og0 = np.array(p0['input']), np.array(p0['output'])
    ih,iw    = ig0.shape
    oh_,ow_  = og0.shape
    if oh_ > ih or ow_ > iw: return None
    for r0 in range(ih-oh_+1):
        for c0 in range(iw-ow_+1):
            if np.array_equal(ig0[r0:r0+oh_, c0:c0+ow_], og0):
                if all(np.array_equal(
                       np.array(p['input'])[r0:r0+oh_, c0:c0+ow_],
                       np.array(p['output'])) for p in pairs[1:]):
                    return r0, c0, oh_, ow_
    return None

def detect_const(pairs):
    if len(set(str(p['output']) for p in pairs)) == 1:
        return g2t(pairs[0]['output'])[np.newaxis]
    return None

def detect_pixel_permutation(pairs):
    ok, sz = _all_same_size(pairs)
    if not ok or sz[0] != sz[1]: return None
    ih, iw = sz[0]
    if ih != H or iw != W: return None
    src = np.array(pairs[0]['input']).flatten()
    dst = np.array(pairs[0]['output']).flatten()
    mapping = {}
    for di, dv in enumerate(dst):
        cands = np.where(src == dv)[0]
        if len(cands) != 1: return None
        mapping[di] = cands[0]
    idx = np.array([mapping[i] for i in range(H*W)], np.int64)
    for p in pairs[1:]:
        s = np.array(p['input']).flatten()
        if not np.array_equal(s[idx], np.array(p['output']).flatten()): return None
    return idx

def detect_roll(pairs):
    ok, sz = _all_same_size(pairs)
    if not ok or sz[0] != sz[1]: return None
    ih, iw = sz[0]
    ig0 = np.array(pairs[0]['input'])
    og0 = np.array(pairs[0]['output'])
    for dr in range(ih):
        for dc in range(iw):
            if dr == 0 and dc == 0: continue
            shifted = np.roll(np.roll(ig0, dr, axis=0), dc, axis=1)
            if np.array_equal(shifted, og0):
                if all(np.array_equal(
                       np.roll(np.roll(np.array(p['input']), dr, 0), dc, 1),
                       np.array(p['output'])) for p in pairs[1:]):
                    return dr, dc, ih, iw
    return None

_GEOM_FNS = [
    ('rot90',   lambda g: np.rot90(g,1),              lambda: _perm_from_fn(lambda r,c:(c,H-1-r))),
    ('rot180',  lambda g: np.rot90(g,2),              lambda: _perm_from_fn(lambda r,c:(H-1-r,W-1-c))),
    ('rot270',  lambda g: np.rot90(g,3),              lambda: _perm_from_fn(lambda r,c:(W-1-c,r))),
    ('hflip',   lambda g: np.fliplr(g),               lambda: _perm_from_fn(lambda r,c:(r,W-1-c))),
    ('vflip',   lambda g: np.flipud(g),               lambda: _perm_from_fn(lambda r,c:(H-1-r,c))),
    ('hvflip',  lambda g: np.flipud(np.fliplr(g)),    lambda: _perm_from_fn(lambda r,c:(H-1-r,W-1-c))),
    ('transp',  lambda g: g.T,                        lambda: _perm_from_fn(lambda r,c:(c,r))),
    ('atransp', lambda g: np.fliplr(np.flipud(g)).T,  lambda: _perm_from_fn(lambda r,c:(W-1-c,H-1-r))),
]

def detect_geom_remap(pairs):
    ok, sz = _all_same_size(pairs)
    if not ok or sz[0] != sz[1]: return None
    for name, geom_fn, idx_fn in _GEOM_FNS:
        faux, valid = [], True
        for p in pairs:
            ig = np.array(p['input']); og = np.array(p['output'])
            if ig.shape != og.shape: valid = False; break
            try: tr_ig = geom_fn(ig)
            except: valid = False; break
            if tr_ig.shape != og.shape: valid = False; break
            faux.append({'input': tr_ig.tolist(), 'output': og.tolist()})
        if not valid: continue
        cmap = detect_color_remap(faux)
        if cmap and any(cmap.get(i,i) != i for i in range(10)):
            return cmap, name, idx_fn()
    return None

# --- NEW v4 DETECTORS ---

def detect_stack_v(pairs):
    p0 = pairs[0]
    ih,iw   = np.array(p0['input']).shape
    oh_,ow_ = np.array(p0['output']).shape
    if iw != ow_: return None
    if oh_ % ih != 0: return None
    n = oh_ // ih
    if n < 2: return None
    for p in pairs:
        ig,og = np.array(p['input']), np.array(p['output'])
        if ig.shape != (ih,iw) or og.shape != (oh_,ow_): return None
        if not np.array_equal(np.tile(ig,(n,1)), og): return None
    return ih, iw, n

def detect_stack_h(pairs):
    p0 = pairs[0]
    ih,iw   = np.array(p0['input']).shape
    oh_,ow_ = np.array(p0['output']).shape
    if ih != oh_: return None
    if ow_ % iw != 0: return None
    n = ow_ // iw
    if n < 2: return None
    for p in pairs:
        ig,og = np.array(p['input']), np.array(p['output'])
        if ig.shape != (ih,iw) or og.shape != (oh_,ow_): return None
        if not np.array_equal(np.tile(ig,(1,n)), og): return None
    return ih, iw, n

def detect_mirror_h(pairs):
    p0 = pairs[0]
    ih,iw   = np.array(p0['input']).shape
    oh_,ow_ = np.array(p0['output']).shape
    if ih != oh_ or ow_ != 2*iw: return None
    for p in pairs:
        ig,og = np.array(p['input']), np.array(p['output'])
        if ig.shape != (ih,iw) or og.shape != (oh_,ow_): return None
        expected = np.concatenate([ig, np.fliplr(ig)], axis=1)
        if not np.array_equal(expected, og): return None
    return ih, iw

def detect_mirror_v(pairs):
    p0 = pairs[0]
    ih,iw   = np.array(p0['input']).shape
    oh_,ow_ = np.array(p0['output']).shape
    if iw != ow_ or oh_ != 2*ih: return None
    for p in pairs:
        ig,og = np.array(p['input']), np.array(p['output'])
        if ig.shape != (ih,iw) or og.shape != (oh_,ow_): return None
        expected = np.concatenate([ig, np.flipud(ig)], axis=0)
        if not np.array_equal(expected, og): return None
    return ih, iw

def detect_single_color(pairs):
    '''All pairs have same solid-color output at same size.'''
    p0 = pairs[0]
    og0 = np.array(p0['output'])
    oh_,ow_ = og0.shape
    if og0.size == 0: return None
    colors = set(og0.flatten().tolist())
    if len(colors) != 1: return None
    color = og0[0,0]
    for p in pairs:
        og = np.array(p['output'])
        if og.shape != (oh_,ow_): return None
        if not np.all(og == color): return None
    return int(color), oh_, ow_

def detect_color_filter(pairs):
    '''Output == input but some colors replaced with 0 (black).'''
    ok, sz = _all_same_size(pairs)
    if not ok or sz[0] != sz[1]: return None
    keep = None
    for p in pairs:
        ig,og = np.array(p['input']), np.array(p['output'])
        if ig.shape != og.shape: return None
        # At each cell, either og == ig OR og == 0
        mask_same = (og == ig)
        mask_zero = (og == 0)
        if not np.all(mask_same | mask_zero): return None
        kept_here = set(ig[mask_same].tolist())
        removed_here = set(ig[(~mask_same) & mask_zero].tolist())
        if keep is None:
            keep = kept_here
            removed_global = removed_here
        else:
            # keep must include all kept-here (some colors may be absent in this pair)
            if kept_here & removed_global: return None
            if removed_here & keep: return None
            keep |= kept_here
    if keep is None or 0 not in keep: keep = keep or set()
    keep.add(0)  # keeping 0 (black) is free -- already zero
    # Must be non-trivial: at least one color removed in some pair
    all_colors_in = set()
    for p in pairs:
        all_colors_in |= set(np.array(p['input']).flatten().tolist())
    removed = all_colors_in - keep
    if not removed: return None
    return sorted(keep)

print('All detectors ready (+ 5 new for size-changing tasks).')
""")

code("""# --- Master solver ---

def try_accept(model, all_p, tier, scratch_path):
    '''Validate + dry-run a candidate model. Returns (model, tier) or (None, None).'''
    # Save to check file size + run onnxruntime
    save_onnx(model, scratch_path)
    if Path(scratch_path).stat().st_size > MAX_BYTES:
        return None, None
    ok, reason = validate_onnx(scratch_path)
    if not ok:
        return None, None
    if not check_all(scratch_path, all_p):
        return None, None
    return model, tier

def solve_task(task_data, task_num, scratch_path):
    train  = task_data.get('train',   [])
    test   = task_data.get('test',    [])
    arcgen = task_data.get('arc-gen', [])
    all_p  = train + test + arcgen
    tr_p   = train + test
    if not all_p: return None, None

    # Tier A: same-size exact transformations
    if detect_identity(tr_p):
        r = try_accept(model_identity(), all_p, 'identity', scratch_path)
        if r[0]: return r

    cmap = detect_color_remap(tr_p)
    if cmap:
        r = try_accept(model_color_remap(cmap), all_p, 'color_remap', scratch_path)
        if r[0]: return r

    rot = detect_rotation(tr_p)
    if rot:
        mfn = {90:model_rot90, 180:model_rot180, 270:model_rot270}[rot]
        r = try_accept(mfn(), all_p, f'rot{rot}', scratch_path)
        if r[0]: return r

    fl = detect_flip(tr_p)
    if fl:
        mfn = {'hflip':model_hflip, 'vflip':model_vflip, 'hvflip':model_hvflip}[fl]
        r = try_accept(mfn(), all_p, fl, scratch_path)
        if r[0]: return r

    if detect_transpose(tr_p):
        r = try_accept(model_transpose(), all_p, 'transpose', scratch_path)
        if r[0]: return r
    if detect_antitranspose(tr_p):
        r = try_accept(model_antitranspose(), all_p, 'antitranspose', scratch_path)
        if r[0]: return r

    # Tier B: size-changing transformations (new in v4)
    sc = detect_scale(tr_p)
    if sc:
        r = try_accept(model_scale(sc), all_p, f'scale{sc}x', scratch_path)
        if r[0]: return r

    tile = detect_tile(tr_p)
    if tile:
        r = try_accept(model_tile(*tile), all_p, f'tile{tile[:2]}', scratch_path)
        if r[0]: return r

    sv = detect_stack_v(tr_p)
    if sv:
        r = try_accept(model_stack_v(*sv), all_p, f'stack_v{sv[2]}', scratch_path)
        if r[0]: return r

    sh = detect_stack_h(tr_p)
    if sh:
        r = try_accept(model_stack_h(*sh), all_p, f'stack_h{sh[2]}', scratch_path)
        if r[0]: return r

    mh = detect_mirror_h(tr_p)
    if mh:
        r = try_accept(model_mirror_h(*mh), all_p, 'mirror_h', scratch_path)
        if r[0]: return r

    mv = detect_mirror_v(tr_p)
    if mv:
        r = try_accept(model_mirror_v(*mv), all_p, 'mirror_v', scratch_path)
        if r[0]: return r

    crop = detect_crop(tr_p)
    if crop:
        r = try_accept(model_crop(*crop), all_p, f'crop{crop[:2]}', scratch_path)
        if r[0]: return r

    roll = detect_roll(tr_p)
    if roll:
        r = try_accept(model_roll(*roll), all_p, f'roll{roll[:2]}', scratch_path)
        if r[0]: return r

    perm = detect_pixel_permutation(tr_p)
    if perm is not None:
        r = try_accept(_gather_model(perm), all_p, 'pixel_perm', scratch_path)
        if r[0]: return r

    # Single-color output (cheaper than const)
    sc1 = detect_single_color(tr_p)
    if sc1:
        color, oh_, ow_ = sc1
        r = try_accept(model_single_color(color, oh_, ow_), all_p, f'single_c{color}', scratch_path)
        if r[0]: return r

    ct = detect_const(tr_p)
    if ct is not None:
        r = try_accept(model_const(ct), all_p, 'const_output', scratch_path)
        if r[0]: return r

    cf = detect_color_filter(tr_p)
    if cf is not None:
        r = try_accept(model_color_filter(cf), all_p, f'color_filter{cf}', scratch_path)
        if r[0]: return r

    gr = detect_geom_remap(tr_p)
    if gr:
        cmap2, name2, idx2 = gr
        r = try_accept(model_remap_geom(cmap2, idx2), all_p, f'remap+{name2}', scratch_path)
        if r[0]: return r

    lin = try_linear_solve(all_p)
    if lin:
        r = try_accept(lin, all_p, 'linear', scratch_path)
        if r[0]: return r

    # Tier C: trained CNN (same-size only)
    sizes = set()
    for p in tr_p:
        ih,iw   = np.array(p['input']).shape
        oh_,ow_ = np.array(p['output']).shape
        sizes.add((ih,iw,oh_,ow_))
    if len(sizes) == 1:
        ih,iw,oh_,ow_ = next(iter(sizes))
        if ih == oh_ and iw == ow_:
            t_cnn = time.time()
            for arch_name, arch_kw in ARCH_LIST:
                remaining = CNN_TIMEOUT - (time.time()-t_cnn)
                if remaining < 1.0: break
                model_t, success = train_conv(train, arch_kw, timeout=remaining)
                if not success: continue
                try:
                    _onnx_export(model_t, scratch_path)
                    if Path(scratch_path).stat().st_size > MAX_BYTES: continue
                    ok, _ = validate_onnx(scratch_path)
                    if not ok: continue
                    passed, total = run_onnx(scratch_path, all_p)
                    if passed == total:
                        m = onnx.load(str(scratch_path))
                        return m, f'cnn_{arch_name}'
                except Exception: pass

    return None, None

print('Master solver ready.')
""")

code("""# --- Run on all 400 tasks ---

task_files = sorted(TASK_DIR.glob('task*.json'))
print(f'Found {len(task_files)} tasks in {TASK_DIR}')

results, solved, tot_score = [], 0, 0.0
t_start = time.time()
SCRATCH = SUB_DIR / '_scratch.onnx'

for tf in task_files:
    task_num = int(tf.stem.replace('task',''))
    t0 = time.time()
    try:
        task_data   = load_task(tf)
        model, tier = solve_task(task_data, task_num, SCRATCH)
    except Exception as e:
        model, tier = None, None
    dt = time.time() - t0

    if model is not None:
        out_path = SUB_DIR / f'task{task_num:03d}.onnx'
        save_onnx(model, out_path)
        # Double-check the saved file
        ok, reason = validate_onnx(out_path)
        if not ok:
            # Drop it -- would error on Kaggle
            out_path.unlink(missing_ok=True)
            model, tier = None, None
            tag = f'XX dropped:{reason[:20]:20s}  {dt:.1f}s'
        else:
            sc = score_model(out_path, verbose=(task_num<=3))
            solved += 1; tot_score += sc
            tag = f'OK {tier[:16]:16s}  {sc:.1f}pts  {dt:.1f}s'
    else:
        tag = f'-- unsolved                  {dt:.1f}s'

    results.append(dict(task=task_num, tier=tier, solved=(model is not None)))

    if model is not None or task_num % 10 == 0:
        elapsed = time.time()-t_start
        eta = elapsed/task_num*(400-task_num) if task_num > 0 else 0
        print(f'[{task_num:3d}] {tag}  | {solved}/{task_num} solved  total={tot_score:.1f}  ETA={eta/60:.1f}m')

# cleanup
if SCRATCH.exists(): SCRATCH.unlink()

elapsed = time.time()-t_start
print(f'\\n{"="*65}')
print(f'Solved: {solved}/400  Score: {tot_score:.1f}  Time: {elapsed:.0f}s ({elapsed/60:.1f}m)')
print(f'{"="*65}')
""")

code("""from collections import Counter
tier_counts = Counter(r['tier'] for r in results if r['tier'])
print('Solutions by tier:')
for t, c in tier_counts.most_common(30):
    print(f'  {t:25s}: {c}')

onnx_files = sorted(SUB_DIR.glob('task*.onnx'))
print(f'\\n{len(onnx_files)} ONNX files in submission')

# Build submission zip
zip_path = OUTPUT_DIR / 'submission.zip'
if zip_path.exists(): zip_path.unlink()
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in onnx_files:
        zf.write(f, arcname=f.name)
print(f'Wrote {zip_path}  ({zip_path.stat().st_size/1024:.1f} KB, {len(onnx_files)} files)')
print(f'\\nDone -- submit {zip_path}')
""")

# Write notebook
nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'neurogolf-v4.ipynb')
with open(out_path, 'w') as f:
    json.dump(nb, f, indent=1)
print(f'Wrote {out_path} with {len(CELLS)} cells')
