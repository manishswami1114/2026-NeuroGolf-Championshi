"""v8 task analyzer: categorize all 400 tasks by their transformation family.

Detects which tasks fall into well-known patterns we can solve analytically,
which is the foundation of reaching high LB.  Prints a histogram + gives lists
of task numbers per category for targeted implementation.
"""
import json, os
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

TASK_DIR = Path(__file__).parent

def load_task(n):
    with open(TASK_DIR / f'task{n:03d}.json') as f:
        return json.load(f)

def train_pairs(t):
    return t.get('train', []) or []

# --- detectors (same transform works on all training pairs) ---

def is_identity(pairs):
    return all(np.array_equal(np.array(p['input']), np.array(p['output'])) for p in pairs)

def is_color_remap(pairs):
    # same shape, consistent 1-to-1 color map
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

def is_rot90(pairs):
    for k in (1, 2, 3):
        if all(np.array_equal(np.rot90(np.array(p['input']), k), np.array(p['output'])) for p in pairs):
            return f'rot{k*90}'
    return None

def is_transpose(pairs):
    if all(np.array_equal(np.array(p['input']).T, np.array(p['output'])) for p in pairs):
        return 'transpose'
    g = lambda x: np.fliplr(np.flipud(x)).T
    if all(np.array_equal(g(np.array(p['input'])), np.array(p['output'])) for p in pairs):
        return 'antitranspose'
    return None

def is_tile(pairs):
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        h, w = ig.shape; oh, ow = og.shape
        if oh % h or ow % w: return None
    # try common tile factors
    for tr in (1,2,3,4,5):
        for tc in (1,2,3,4,5):
            if (tr,tc)==(1,1): continue
            if all(np.array_equal(np.tile(np.array(p['input']),(tr,tc)), np.array(p['output'])) for p in pairs):
                return f'tile{tr}x{tc}'
    return None

def is_crop_bbox(pairs):
    # output is bounding box of non-zero region of input
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        nz = np.argwhere(ig != 0)
        if len(nz)==0: return False
        r0,c0 = nz.min(0); r1,c1 = nz.max(0)
        if not np.array_equal(ig[r0:r1+1, c0:c1+1], og): return False
    return True

def is_crop_majority_color(pairs):
    # output = bbox of the most-common non-bg color
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

def is_mirror_h(pairs):
    # concat(input, hflip(input)) on columns
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        if not np.array_equal(np.concatenate([ig, np.fliplr(ig)], axis=1), og): return False
    return True

def is_mirror_v(pairs):
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        if not np.array_equal(np.concatenate([ig, np.flipud(ig)], axis=0), og): return False
    return True

def is_fractal_self_tile(pairs):
    # output[r*h:(r+1)*h, c*w:(c+1)*w] = input if input[r,c] != 0 else zeros
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
    # Alternate fractal: output tile = (ig if ig[r,c]==key else zeros)
    # Where key is a specific foreground color
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        h, w = ig.shape
        if og.shape != (h*h, w*w): return False
        # try: for each tile, compare to input and check tile is either ig or zeros
        for r in range(h):
            for c in range(w):
                tile = og[r*h:(r+1)*h, c*w:(c+1)*w]
                if not (np.array_equal(tile, ig) or np.array_equal(tile, np.zeros_like(ig))):
                    return False
    return True

def is_repeat_rows(pairs):
    # Output = rows of input each repeated k times
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        h, w = ig.shape; oh, ow = og.shape
        if w != ow or oh % h: return None
        k = oh // h
        if not np.array_equal(np.repeat(ig, k, axis=0), og): return None
    return True

def is_repeat_cols(pairs):
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        h, w = ig.shape; oh, ow = og.shape
        if h != oh or ow % w: return None
        k = ow // w
        if not np.array_equal(np.repeat(ig, k, axis=1), og): return None
    return True

def is_gravity_down(pairs):
    # Each column: non-zero pixels fall to bottom preserving order
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        if ig.shape != og.shape: return False
        H, W = ig.shape
        for c in range(W):
            col = ig[:, c]
            nz = [v for v in col if v != 0]
            expected = np.concatenate([np.zeros(H-len(nz), int), np.array(nz, int)])
            if not np.array_equal(expected, og[:, c]): return False
    return True

def is_gravity_up(pairs):
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        if ig.shape != og.shape: return False
        H, W = ig.shape
        for c in range(W):
            col = ig[:, c]
            nz = [v for v in col if v != 0]
            expected = np.concatenate([np.array(nz, int), np.zeros(H-len(nz), int)])
            if not np.array_equal(expected, og[:, c]): return False
    return True

def is_gravity_left(pairs):
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        if ig.shape != og.shape: return False
        H, W = ig.shape
        for r in range(H):
            row = ig[r, :]
            nz = [v for v in row if v != 0]
            expected = np.concatenate([np.array(nz, int), np.zeros(W-len(nz), int)])
            if not np.array_equal(expected, og[r, :]): return False
    return True

def is_gravity_right(pairs):
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        if ig.shape != og.shape: return False
        H, W = ig.shape
        for r in range(H):
            row = ig[r, :]
            nz = [v for v in row if v != 0]
            expected = np.concatenate([np.zeros(W-len(nz), int), np.array(nz, int)])
            if not np.array_equal(expected, og[r, :]): return False
    return True

def is_row_sort(pairs):
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        if ig.shape != og.shape: return False
        for r in range(ig.shape[0]):
            if not np.array_equal(np.sort(ig[r]), og[r]) and not np.array_equal(np.sort(ig[r])[::-1], og[r]):
                return False
    return True

def is_same_shape(pairs):
    return all(np.array(p['input']).shape == np.array(p['output']).shape for p in pairs)

def is_fixed_output_shape(pairs):
    shapes = set(tuple(np.array(p['output']).shape) for p in pairs)
    return len(shapes) == 1

def is_bg_fill(pairs):
    # output = replace a specific color in input with another specific color
    # (like color_remap but shape-preserving)
    return is_same_shape(pairs) and is_color_remap(pairs)

def is_scale_up(pairs, k):
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        expected = np.kron(ig, np.ones((k,k), int))
        if not np.array_equal(expected, og): return False
    return True

# --- Main analysis ---

categories = defaultdict(list)  # category_name -> [task_nums]
shape_info = defaultdict(int)

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

    # try each detector (earliest match wins to keep categories clean)
    tag = None
    if is_identity(pairs):            tag = 'identity'
    elif is_const_output(pairs):      tag = 'const_output'
    elif is_hflip(pairs):             tag = 'hflip'
    elif is_vflip(pairs):             tag = 'vflip'
    elif is_rot90(pairs):             tag = is_rot90(pairs)
    elif is_transpose(pairs):         tag = is_transpose(pairs)
    elif is_color_remap(pairs):       tag = 'color_remap'
    elif is_mirror_h(pairs):          tag = 'mirror_h'
    elif is_mirror_v(pairs):          tag = 'mirror_v'
    elif is_fractal_self_tile(pairs): tag = 'fractal_self_tile'
    elif is_fractal_cell_is_color(pairs): tag = 'fractal_cell_is_color'
    elif is_tile(pairs):              tag = is_tile(pairs)
    elif is_scale_up(pairs, 2):       tag = 'scale_up_2x'
    elif is_scale_up(pairs, 3):       tag = 'scale_up_3x'
    elif is_scale_up(pairs, 4):       tag = 'scale_up_4x'
    elif is_repeat_rows(pairs):       tag = 'repeat_rows'
    elif is_repeat_cols(pairs):       tag = 'repeat_cols'
    elif is_gravity_down(pairs):      tag = 'gravity_down'
    elif is_gravity_up(pairs):        tag = 'gravity_up'
    elif is_gravity_left(pairs):      tag = 'gravity_left'
    elif is_gravity_right(pairs):     tag = 'gravity_right'
    elif is_crop_bbox(pairs):         tag = 'crop_bbox'
    elif is_crop_majority_color(pairs): tag = 'crop_majority_color'
    elif is_row_sort(pairs):          tag = 'row_sort'
    else:                              tag = 'UNKNOWN'

    categories[tag].append(n)

    # shape info
    if is_same_shape(pairs):
        shape_info['same_shape'] += 1
    if is_fixed_output_shape(pairs):
        shape_info['fixed_out_shape'] += 1

# Report
print(f"=== Transformation family histogram across 400 tasks ===\n")
total_known = 0
for name, tasks in sorted(categories.items(), key=lambda kv: -len(kv[1])):
    if name == 'UNKNOWN': continue
    total_known += len(tasks)
    print(f"  {name:28s} {len(tasks):3d}  examples: {tasks[:5]}")

unknown = categories.get('UNKNOWN', [])
print(f"\n  {'UNKNOWN':28s} {len(unknown):3d}  examples: {unknown[:10]}")
print(f"\n=== Known: {total_known}/400  Unknown: {len(unknown)}/400 ===")
print(f"\nShape info: same_shape={shape_info['same_shape']}  fixed_out_shape={shape_info['fixed_out_shape']}")

# Save UNKNOWN list for next-step inspection
with open(TASK_DIR / '_unknown_tasks.txt', 'w') as f:
    for n in unknown:
        f.write(f"{n}\n")
print(f"\nWrote _unknown_tasks.txt ({len(unknown)} tasks) for manual inspection")
