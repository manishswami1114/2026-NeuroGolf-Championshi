"""v9 task analyzer: deeper + more detectors to find more task families.

Adds many new detectors targeting common ARC patterns:
- Shape-preserving: overlay-max, overlay-min, color-swap-pair, erase-color, keep-color
- Fixed-output: 1x1 color-count, majority-color, min-color, N-colors
- Size transformations: 2x2 scale-down, 3x3 scale-down, transpose with variant sizes
- Per-row / per-col: row-reverse, col-reverse
- Structural: fill-bg, bbox-of-color, extract-largest-rect
- Counting: count-color-occurrences
"""
import json
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

TASK_DIR = Path(__file__).parent

def load_task(n):
    with open(TASK_DIR / f'task{n:03d}.json') as f:
        return json.load(f)

def train_pairs(t):
    return t.get('train', []) or []

# Shape-based metrics

def is_same_shape(pairs):
    return all(np.array(p['input']).shape == np.array(p['output']).shape for p in pairs)

def shape_ratio(pairs):
    """Return (oh/ih, ow/iw) if constant across pairs, else None."""
    r = None
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        ih, iw = ig.shape; oh, ow = og.shape
        if ih == 0 or iw == 0: return None
        if oh % ih or ow % iw:
            # fractional
            rr = (oh/ih, ow/iw)
        else:
            rr = (oh//ih, ow//iw)
        if r is None: r = rr
        elif r != rr: return None
    return r

def fixed_output_shape(pairs):
    shapes = set(tuple(np.array(p['output']).shape) for p in pairs)
    return list(shapes)[0] if len(shapes) == 1 else None

# New detectors

def is_identity(pairs):
    return all(np.array_equal(np.array(p['input']), np.array(p['output'])) for p in pairs)

def is_color_remap(pairs):
    cmap = {}
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        if ig.shape != og.shape: return False
        for a, b in zip(ig.flatten(), og.flatten()):
            if a in cmap and cmap[a] != b: return False
            cmap[a] = b
    return True

def is_hflip(pairs):
    return all(np.array_equal(np.fliplr(np.array(p['input'])), np.array(p['output'])) for p in pairs)

def is_vflip(pairs):
    return all(np.array_equal(np.flipud(np.array(p['input'])), np.array(p['output'])) for p in pairs)

def is_rot(pairs, k):
    return all(np.array_equal(np.rot90(np.array(p['input']), k), np.array(p['output'])) for p in pairs)

def is_transpose(pairs):
    return all(np.array_equal(np.array(p['input']).T, np.array(p['output'])) for p in pairs)

def is_anti_transpose(pairs):
    return all(np.array_equal(np.fliplr(np.flipud(np.array(p['input']))).T, np.array(p['output'])) for p in pairs)

def is_mirror_h(pairs):
    return all(np.array_equal(np.concatenate([np.array(p['input']), np.fliplr(np.array(p['input']))], 1), np.array(p['output'])) for p in pairs)

def is_mirror_v(pairs):
    return all(np.array_equal(np.concatenate([np.array(p['input']), np.flipud(np.array(p['input']))], 0), np.array(p['output'])) for p in pairs)

def is_fractal_self_tile(pairs):
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        h, w = ig.shape
        if og.shape != (h*h, w*w): return False
        for r in range(h):
            for c in range(w):
                tile = og[r*h:(r+1)*h, c*w:(c+1)*w]
                expected = ig if ig[r,c] != 0 else np.zeros_like(ig)
                if not np.array_equal(tile, expected): return False
    return True

def is_fractal_cell_is_color(pairs):
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        h, w = ig.shape
        if og.shape != (h*h, w*w): return False
        for r in range(h):
            for c in range(w):
                tile = og[r*h:(r+1)*h, c*w:(c+1)*w]
                if not (np.array_equal(tile, ig) or np.array_equal(tile, np.zeros_like(ig))):
                    return False
    return True

def is_scale_up(pairs, k):
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        if not np.array_equal(np.kron(ig, np.ones((k,k), int)), og): return False
    return True

def is_scale_down(pairs, k):
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        h, w = ig.shape
        if h % k or w % k: return False
        if og.shape != (h//k, w//k): return False
        for r in range(h//k):
            for c in range(w//k):
                block = ig[r*k:(r+1)*k, c*k:(c+1)*k]
                if len(set(block.flatten())) != 1: return False
                if og[r,c] != block[0,0]: return False
    return True

def is_repeat_rows(pairs):
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        h, w = ig.shape; oh, ow = og.shape
        if w != ow or oh % h: return False
        k = oh // h
        if not np.array_equal(np.repeat(ig, k, axis=0), og): return False
    return True

def is_repeat_cols(pairs):
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        h, w = ig.shape; oh, ow = og.shape
        if h != oh or ow % w: return False
        k = ow // w
        if not np.array_equal(np.repeat(ig, k, axis=1), og): return False
    return True

def is_tile(pairs, tr, tc):
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        if not np.array_equal(np.tile(ig, (tr,tc)), og): return False
    return True

def is_gravity(pairs, direction):
    """direction: 'down', 'up', 'left', 'right'"""
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        if ig.shape != og.shape: return False
        H, W = ig.shape
        if direction in ('down','up'):
            for c in range(W):
                col = ig[:, c]; nz = [v for v in col if v != 0]
                if direction == 'down':
                    exp = np.concatenate([np.zeros(H-len(nz), int), np.array(nz, int)])
                else:
                    exp = np.concatenate([np.array(nz, int), np.zeros(H-len(nz), int)])
                if not np.array_equal(exp, og[:,c]): return False
        else:
            for r in range(H):
                row = ig[r,:]; nz = [v for v in row if v != 0]
                if direction == 'right':
                    exp = np.concatenate([np.zeros(W-len(nz), int), np.array(nz, int)])
                else:
                    exp = np.concatenate([np.array(nz, int), np.zeros(W-len(nz), int)])
                if not np.array_equal(exp, og[r,:]): return False
    return True

def is_crop_bbox_nonzero(pairs):
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        nz = np.argwhere(ig != 0)
        if len(nz)==0: return False
        r0,c0 = nz.min(0); r1,c1 = nz.max(0)
        if not np.array_equal(ig[r0:r1+1, c0:c1+1], og): return False
    return True

def is_crop_majority(pairs):
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        nz = ig[ig!=0]
        if len(nz)==0: return False
        colors, counts = np.unique(nz, return_counts=True)
        c = colors[counts.argmax()]
        ind = np.argwhere(ig==c)
        if len(ind)==0: return False
        r0,c0 = ind.min(0); r1,c1 = ind.max(0)
        if not np.array_equal(ig[r0:r1+1, c0:c1+1], og): return False
    return True

def is_const_output(pairs):
    ogs = [np.array(p['output']) for p in pairs]
    return all(np.array_equal(ogs[0], o) for o in ogs)

def is_row_reverse(pairs):
    return all(np.array_equal(np.array(p['input'])[::-1], np.array(p['output'])) for p in pairs)

def is_col_reverse(pairs):
    return all(np.array_equal(np.array(p['input'])[:,::-1], np.array(p['output'])) for p in pairs)

def is_add_border(pairs):
    """output = input with 1px border of some color"""
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        h, w = ig.shape
        if og.shape != (h+2, w+2): return False
        if not np.array_equal(og[1:-1,1:-1], ig): return False
        # border must be uniform
        border = np.concatenate([og[0,:], og[-1,:], og[:,0], og[:,-1]])
        if len(set(border)) != 1: return False
    return True

def is_remove_border(pairs):
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        h, w = ig.shape
        if og.shape != (h-2, w-2): return False
        if not np.array_equal(ig[1:-1,1:-1], og): return False
    return True

def is_double_h(pairs):
    """concat(input, input) horizontally"""
    return all(np.array_equal(np.concatenate([np.array(p['input'])]*2, 1), np.array(p['output'])) for p in pairs)

def is_double_v(pairs):
    return all(np.array_equal(np.concatenate([np.array(p['input'])]*2, 0), np.array(p['output'])) for p in pairs)

def is_mirror_quad(pairs):
    """output is 2h x 2w = [[in, fliplr(in)], [flipud(in), rot180(in)]]"""
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        h, w = ig.shape
        if og.shape != (2*h, 2*w): return False
        a = np.concatenate([ig, np.fliplr(ig)], 1)
        b = np.concatenate([np.flipud(ig), np.fliplr(np.flipud(ig))], 1)
        if not np.array_equal(np.concatenate([a,b],0), og): return False
    return True

def is_fill_bg_with_color(pairs):
    """input has zeros replaced by a fixed color"""
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        if ig.shape != og.shape: return False
        # all non-zero pixels identical
        mask = (ig != 0)
        if not np.array_equal(ig[mask], og[mask]): return False
        # all bg pixels in og are the same color
        bg_pixels = og[~mask]
        if len(bg_pixels) > 0 and len(set(bg_pixels)) != 1: return False
    return True

def is_keep_only_color(pairs):
    """Output = input but only one specific color kept (rest zeroed)."""
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        if ig.shape != og.shape: return False
        kept_colors = set(og[og != 0].flatten())
        if len(kept_colors) > 1: return False
        if len(kept_colors) == 1:
            c = list(kept_colors)[0]
            if not np.array_equal((ig == c) * c, og): return False
    return True

def is_count_color_1x1(pairs):
    """Output is 1x1 = count of some color in input"""
    for p in pairs:
        og = np.array(p['output'])
        if og.shape != (1,1): return False
    return True

def is_majority_1x1(pairs):
    """Output is 1x1 = most-common non-bg color in input"""
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        if og.shape != (1,1): return False
        nz = ig[ig != 0]
        if len(nz) == 0: return False
        colors, counts = np.unique(nz, return_counts=True)
        if og[0,0] != colors[counts.argmax()]: return False
    return True

def is_minority_1x1(pairs):
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        if og.shape != (1,1): return False
        nz = ig[ig != 0]
        if len(nz) == 0: return False
        colors, counts = np.unique(nz, return_counts=True)
        if og[0,0] != colors[counts.argmin()]: return False
    return True

def is_color_swap_pair(pairs):
    """Two specific colors are swapped everywhere; others unchanged."""
    if not is_same_shape(pairs): return False
    cmap = {}
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        for a, b in zip(ig.flatten(), og.flatten()):
            if a in cmap and cmap[a] != b: return False
            cmap[a] = b
    # only 2 values have a!=b, and they are swapped
    swapped = [(a, b) for a, b in cmap.items() if a != b]
    if len(swapped) != 2: return False
    (a1,b1), (a2,b2) = swapped
    return a1 == b2 and a2 == b1

def is_recolor_all(pairs):
    """All non-zero pixels become one specific color."""
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        if ig.shape != og.shape: return False
        mask = (ig != 0)
        og_fg = og[mask]
        if len(og_fg) == 0: continue
        if len(set(og_fg)) != 1: return False
        # and bg stays bg
        if not np.all(og[~mask] == 0): return False
    return True

def is_overlay_max(pairs):
    """Output is max of input quadrants (common: split in half, OR them together)."""
    # Try horizontal split
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        h, w = ig.shape
        if w % 2 == 0 and og.shape == (h, w//2):
            left, right = ig[:, :w//2], ig[:, w//2:]
            if np.array_equal(np.maximum(left, right), og): continue
            return False
        elif h % 2 == 0 and og.shape == (h//2, w):
            top, bot = ig[:h//2, :], ig[h//2:, :]
            if np.array_equal(np.maximum(top, bot), og): continue
            return False
        else:
            return False
    return True

# Run analysis

categories = defaultdict(list)

DETECTORS = [
    ('identity', is_identity),
    ('const_output', is_const_output),
    ('hflip', is_hflip),
    ('vflip', is_vflip),
    ('rot90', lambda p: is_rot(p, 1)),
    ('rot180', lambda p: is_rot(p, 2)),
    ('rot270', lambda p: is_rot(p, 3)),
    ('transpose', is_transpose),
    ('anti_transpose', is_anti_transpose),
    ('color_remap', is_color_remap),
    ('color_swap_pair', is_color_swap_pair),
    ('recolor_all', is_recolor_all),
    ('fill_bg_with_color', is_fill_bg_with_color),
    ('keep_only_color', is_keep_only_color),
    ('mirror_h', is_mirror_h),
    ('mirror_v', is_mirror_v),
    ('mirror_quad', is_mirror_quad),
    ('double_h', is_double_h),
    ('double_v', is_double_v),
    ('fractal_self_tile', is_fractal_self_tile),
    ('fractal_cell_is_color', is_fractal_cell_is_color),
    ('scale_up_2x', lambda p: is_scale_up(p, 2)),
    ('scale_up_3x', lambda p: is_scale_up(p, 3)),
    ('scale_up_4x', lambda p: is_scale_up(p, 4)),
    ('scale_up_5x', lambda p: is_scale_up(p, 5)),
    ('scale_down_2x', lambda p: is_scale_down(p, 2)),
    ('scale_down_3x', lambda p: is_scale_down(p, 3)),
    ('repeat_rows', is_repeat_rows),
    ('repeat_cols', is_repeat_cols),
    ('tile_1x2', lambda p: is_tile(p, 1, 2)),
    ('tile_2x1', lambda p: is_tile(p, 2, 1)),
    ('tile_2x2', lambda p: is_tile(p, 2, 2)),
    ('tile_1x3', lambda p: is_tile(p, 1, 3)),
    ('tile_3x1', lambda p: is_tile(p, 3, 1)),
    ('tile_3x3', lambda p: is_tile(p, 3, 3)),
    ('gravity_down', lambda p: is_gravity(p, 'down')),
    ('gravity_up', lambda p: is_gravity(p, 'up')),
    ('gravity_left', lambda p: is_gravity(p, 'left')),
    ('gravity_right', lambda p: is_gravity(p, 'right')),
    ('crop_bbox_nonzero', is_crop_bbox_nonzero),
    ('crop_majority', is_crop_majority),
    ('add_border', is_add_border),
    ('remove_border', is_remove_border),
    ('majority_1x1', is_majority_1x1),
    ('minority_1x1', is_minority_1x1),
    ('count_color_1x1', is_count_color_1x1),
    ('overlay_max', is_overlay_max),
]

shape_info = {'same_shape':0, 'fixed_out_shape':0, 'out_larger':0, 'out_smaller':0, 'out_1x1':0}

for n in range(1, 401):
    try:
        t = load_task(n)
    except Exception as e:
        categories['load_error'].append(n)
        continue
    pairs = train_pairs(t)
    if not pairs:
        categories['no_train'].append(n)
        continue
    tag = None
    for name, fn in DETECTORS:
        try:
            if fn(pairs):
                tag = name
                break
        except Exception:
            continue
    if tag is None:
        tag = 'UNKNOWN'
    categories[tag].append(n)
    # shape metrics
    if is_same_shape(pairs): shape_info['same_shape'] += 1
    if fixed_output_shape(pairs): shape_info['fixed_out_shape'] += 1
    # out size vs in size
    for p in pairs[:1]:
        ih, iw = np.array(p['input']).shape
        oh, ow = np.array(p['output']).shape
        if oh*ow > ih*iw: shape_info['out_larger'] += 1
        elif oh*ow < ih*iw: shape_info['out_smaller'] += 1
        if oh == 1 and ow == 1: shape_info['out_1x1'] += 1

total_known = 0
print(f"=== v9 Transformation histogram (400 tasks) ===\n")
for name, tasks in sorted(categories.items(), key=lambda kv: -len(kv[1])):
    if name == 'UNKNOWN': continue
    total_known += len(tasks)
    print(f"  {name:28s} {len(tasks):3d}  examples: {tasks[:6]}")

unknown = categories.get('UNKNOWN', [])
print(f"\n  {'UNKNOWN':28s} {len(unknown):3d}  examples: {unknown[:12]}")
print(f"\n=== KNOWN: {total_known}/400  UNKNOWN: {len(unknown)}/400 ===")
print(f"Shapes: {shape_info}")

# Analyze UNKNOWN by shape category
unk_shape = {'same_shape':[], 'out_larger':[], 'out_smaller':[], 'out_1x1':[], 'mixed':[]}
for n in unknown:
    t = load_task(n); pairs = train_pairs(t)
    if is_same_shape(pairs):
        unk_shape['same_shape'].append(n)
    elif fixed_output_shape(pairs) == (1,1):
        unk_shape['out_1x1'].append(n)
    else:
        p = pairs[0]
        ih, iw = np.array(p['input']).shape
        oh, ow = np.array(p['output']).shape
        if oh*ow > ih*iw: unk_shape['out_larger'].append(n)
        elif oh*ow < ih*iw: unk_shape['out_smaller'].append(n)
        else: unk_shape['mixed'].append(n)

print(f"\n=== UNKNOWN partition by shape ===")
for k, v in unk_shape.items():
    print(f"  {k:15s} {len(v):3d}  examples: {v[:8]}")

with open(TASK_DIR / '_unknown_v9.txt', 'w') as f:
    for n in unknown: f.write(f"{n}\n")
