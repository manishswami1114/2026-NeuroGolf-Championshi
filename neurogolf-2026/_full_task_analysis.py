#!/usr/bin/env python3
"""
Full analysis of all 400 NeuroGolf tasks.
For each task: show input/output sizes, colors, and classify the transformation.
Output as a big CSV-like table + summary stats.
"""

import json, os, csv
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path

TASK_DIR = Path("/Users/manishswami/developer/Nuero-golf championship/neurogolf-2026/")
OUT_CSV  = TASK_DIR / "_task_analysis_full.csv"

# ── helpers ──────────────────────────────────────────────────────────────────

def load(path):
    with open(path) as f:
        return json.load(f)

def grid_shape(g):
    a = np.array(g)
    return a.shape

def grid_colors(g):
    return sorted(set(np.array(g).flatten().tolist()))

# ── detectors (run on ALL train pairs; return True only if ALL match) ────────

def is_identity(pairs):
    return all(np.array_equal(np.array(p['input']), np.array(p['output'])) for p in pairs)

def is_color_remap(pairs):
    """Pure per-pixel color substitution (1x1 conv solvable)."""
    cmap = {}
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        if ig.shape != og.shape:
            return False
        for s, d in zip(ig.flatten(), og.flatten()):
            s, d = int(s), int(d)
            if s in cmap:
                if cmap[s] != d:
                    return False
            else:
                cmap[s] = d
    # must actually remap something (not identity)
    return any(k != v for k, v in cmap.items())

def is_rotation(pairs):
    for k in (1, 2, 3):
        if all(np.array_equal(np.rot90(np.array(p['input']), k), np.array(p['output'])) for p in pairs):
            return f"rot{k*90}"
    return None

def is_flip(pairs):
    checks = [
        ("hflip",  lambda g: np.fliplr(g)),
        ("vflip",  lambda g: np.flipud(g)),
        ("hvflip", lambda g: np.flipud(np.fliplr(g))),
    ]
    for name, fn in checks:
        if all(np.array_equal(fn(np.array(p['input'])), np.array(p['output'])) for p in pairs):
            return name
    return None

def is_transpose(pairs):
    if all(np.array_equal(np.array(p['input']).T, np.array(p['output'])) for p in pairs):
        return "transpose"
    if all(np.array_equal(np.fliplr(np.flipud(np.array(p['input']))).T, np.array(p['output'])) for p in pairs):
        return "antitranspose"
    return None

def is_scale(pairs):
    for f in (2, 3, 4, 5, 6):
        if all(np.array_equal(np.repeat(np.repeat(np.array(p['input']), f, 0), f, 1), np.array(p['output'])) for p in pairs):
            return f"scale{f}x"
    return None

def is_scale_up(pairs):
    """Each cell becomes kxk block (kron product)."""
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        ih, iw = ig.shape
        oh, ow = og.shape
        if oh == 0 or ow == 0 or ih == 0 or iw == 0:
            return None
        if oh % ih != 0 or ow % iw != 0:
            return None
        if oh // ih != ow // iw:
            return None
    # consistent k
    ig0, og0 = np.array(pairs[0]['input']), np.array(pairs[0]['output'])
    k = og0.shape[0] // ig0.shape[0]
    if k <= 1:
        return None
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        if not np.array_equal(np.kron(ig, np.ones((k, k), int)), og):
            return None
    return f"scale_up_{k}x"

def is_tile(pairs):
    ig0, og0 = np.array(pairs[0]['input']), np.array(pairs[0]['output'])
    ih, iw = ig0.shape
    oh, ow = og0.shape
    if oh % ih != 0 or ow % iw != 0:
        return None
    tr, tc = oh // ih, ow // iw
    if tr == 1 and tc == 1:
        return None
    if all(np.array_equal(np.tile(np.array(p['input']), (tr, tc)), np.array(p['output'])) for p in pairs):
        return f"tile_{tr}x{tc}"
    return None

def is_crop(pairs):
    ig0, og0 = np.array(pairs[0]['input']), np.array(pairs[0]['output'])
    ih, iw = ig0.shape
    oh, ow = og0.shape
    if oh > ih or ow > iw:
        return None
    for r0 in range(ih - oh + 1):
        for c0 in range(iw - ow + 1):
            if np.array_equal(ig0[r0:r0+oh, c0:c0+ow], og0):
                if all(np.array_equal(np.array(p['input'])[r0:r0+oh, c0:c0+ow], np.array(p['output'])) for p in pairs[1:]):
                    return f"crop_r{r0}c{c0}_{oh}x{ow}"
    return None

def is_const_output(pairs):
    outputs = [str(p['output']) for p in pairs]
    if len(set(outputs)) == 1:
        return "const_output"
    return None

def is_roll(pairs):
    ig0, og0 = np.array(pairs[0]['input']), np.array(pairs[0]['output'])
    if ig0.shape != og0.shape:
        return None
    ih, iw = ig0.shape
    for dr in range(ih):
        for dc in range(iw):
            if dr == 0 and dc == 0:
                continue
            shifted = np.roll(np.roll(ig0, dr, axis=0), dc, axis=1)
            if np.array_equal(shifted, og0):
                if all(np.array_equal(
                    np.roll(np.roll(np.array(p['input']), dr, 0), dc, 1),
                    np.array(p['output'])) for p in pairs[1:]):
                    return f"roll_dr{dr}_dc{dc}"
    return None

def is_mirror_h(pairs):
    ig0, og0 = np.array(pairs[0]['input']), np.array(pairs[0]['output'])
    ih, iw = ig0.shape
    oh, ow = og0.shape
    if ih != oh or ow != 2 * iw:
        return None
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        if ig.shape != (ih, iw) or og.shape != (oh, ow):
            return None
        if not np.array_equal(np.concatenate([ig, np.fliplr(ig)], axis=1), og):
            return None
    return "mirror_h"

def is_mirror_v(pairs):
    ig0, og0 = np.array(pairs[0]['input']), np.array(pairs[0]['output'])
    ih, iw = ig0.shape
    oh, ow = og0.shape
    if iw != ow or oh != 2 * ih:
        return None
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        if ig.shape != (ih, iw) or og.shape != (oh, ow):
            return None
        if not np.array_equal(np.concatenate([ig, np.flipud(ig)], axis=0), og):
            return None
    return "mirror_v"

def is_stack_v(pairs):
    ig0, og0 = np.array(pairs[0]['input']), np.array(pairs[0]['output'])
    ih, iw = ig0.shape
    oh, ow = og0.shape
    if iw != ow or oh % ih != 0:
        return None
    n = oh // ih
    if n < 2:
        return None
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        if ig.shape != (ih, iw) or og.shape != (oh, ow):
            return None
        if not np.array_equal(np.tile(ig, (n, 1)), og):
            return None
    return f"stack_v_{n}"

def is_stack_h(pairs):
    ig0, og0 = np.array(pairs[0]['input']), np.array(pairs[0]['output'])
    ih, iw = ig0.shape
    oh, ow = og0.shape
    if ih != oh or ow % iw != 0:
        return None
    n = ow // iw
    if n < 2:
        return None
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        if ig.shape != (ih, iw) or og.shape != (oh, ow):
            return None
        if not np.array_equal(np.tile(ig, (1, n)), og):
            return None
    return f"stack_h_{n}"

def is_single_color_output(pairs):
    og0 = np.array(pairs[0]['output'])
    oh, ow = og0.shape
    colors = set(og0.flatten().tolist())
    if len(colors) != 1:
        return None
    color = og0.flat[0]
    for p in pairs:
        og = np.array(p['output'])
        if og.shape != (oh, ow):
            return None
        if not np.all(og == color):
            return None
    return f"single_color_{int(color)}"

def is_color_filter(pairs):
    """Output = input but with some colors zeroed out."""
    keep = None
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        if ig.shape != og.shape:
            return None
        mask_same = (og == ig)
        mask_zero = (og == 0)
        if not np.all(mask_same | mask_zero):
            return None
        kept_here = set(ig[mask_same].tolist())
        if keep is None:
            keep = kept_here
        # just check consistency loosely
    if keep is None:
        return None
    return "color_filter"

def is_repeat_rows(pairs):
    for k in (2, 3, 4, 5):
        ok = True
        for p in pairs:
            ig, og = np.array(p['input']), np.array(p['output'])
            ih, iw = ig.shape
            if og.shape != (ih * k, iw):
                ok = False; break
            if not np.array_equal(np.repeat(ig, k, axis=0), og):
                ok = False; break
        if ok:
            return f"repeat_rows_{k}x"
    return None

def is_repeat_cols(pairs):
    for k in (2, 3, 4, 5):
        ok = True
        for p in pairs:
            ig, og = np.array(p['input']), np.array(p['output'])
            ih, iw = ig.shape
            if og.shape != (ih, iw * k):
                ok = False; break
            if not np.array_equal(np.repeat(ig, k, axis=1), og):
                ok = False; break
        if ok:
            return f"repeat_cols_{k}x"
    return None

def is_border(pairs):
    ig0, og0 = np.array(pairs[0]['input']), np.array(pairs[0]['output'])
    ih, iw = ig0.shape
    if og0.shape != (ih + 2, iw + 2):
        return None
    if not np.array_equal(og0[1:-1, 1:-1], ig0):
        return None
    border_pixels = np.concatenate([og0[0, :], og0[-1, :], og0[:, 0], og0[:, -1]])
    if len(set(border_pixels.tolist())) != 1:
        return None
    color = int(og0[0, 0])
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        if og.shape != (ig.shape[0] + 2, ig.shape[1] + 2):
            return None
        if not np.array_equal(og[1:-1, 1:-1], ig):
            return None
    return f"border_c{color}"

def is_fractal(pairs):
    ig0 = np.array(pairs[0]['input'])
    og0 = np.array(pairs[0]['output'])
    ih, iw = ig0.shape
    if og0.shape != (ih * ih, iw * iw):
        return None
    for p in pairs:
        ig, og = np.array(p['input']), np.array(p['output'])
        if ig.shape != (ih, iw) or og.shape != (ih * ih, iw * iw):
            return None
        for r in range(ih):
            for c in range(iw):
                tile = og[r*ih:(r+1)*ih, c*iw:(c+1)*iw]
                expected = ig if ig[r, c] != 0 else np.zeros_like(ig)
                if not np.array_equal(tile, expected):
                    return None
    return "fractal_self_tile"

def is_geom_plus_remap(pairs):
    """Geometric transform + color remap combo."""
    geom_fns = [
        ("rot90+remap",   lambda g: np.rot90(g, 1)),
        ("rot180+remap",  lambda g: np.rot90(g, 2)),
        ("rot270+remap",  lambda g: np.rot90(g, 3)),
        ("hflip+remap",   lambda g: np.fliplr(g)),
        ("vflip+remap",   lambda g: np.flipud(g)),
        ("transp+remap",  lambda g: g.T),
    ]
    for name, fn in geom_fns:
        cmap = {}
        valid = True
        for p in pairs:
            ig, og = np.array(p['input']), np.array(p['output'])
            try:
                tr_ig = fn(ig)
            except:
                valid = False; break
            if tr_ig.shape != og.shape:
                valid = False; break
            for s, d in zip(tr_ig.flatten(), og.flatten()):
                s, d = int(s), int(d)
                if s in cmap:
                    if cmap[s] != d:
                        valid = False; break
                else:
                    cmap[s] = d
            if not valid:
                break
        if valid and cmap and any(k != v for k, v in cmap.items()):
            return name
    return None

# ── size relationship classifier ─────────────────────────────────────────────

def size_relation(pairs):
    """Classify the input→output size relationship."""
    shapes = set()
    for p in pairs:
        ishp = tuple(np.array(p['input']).shape)
        oshp = tuple(np.array(p['output']).shape)
        shapes.add((ishp, oshp))
    
    if len(shapes) == 1:
        (ishp, oshp) = shapes.pop()
        if ishp == oshp:
            return "same_size", f"{ishp[0]}x{ishp[1]}"
        elif ishp[0] < oshp[0] or ishp[1] < oshp[1]:
            return "expand", f"{ishp[0]}x{ishp[1]}→{oshp[0]}x{oshp[1]}"
        else:
            return "shrink", f"{ishp[0]}x{ishp[1]}→{oshp[0]}x{oshp[1]}"
    else:
        # variable sizes across examples
        all_same = all(np.array(p['input']).shape == np.array(p['output']).shape for p in pairs)
        if all_same:
            return "same_size_variable", "varies"
        # check if output is always same shape
        out_shapes = set(tuple(np.array(p['output']).shape) for p in pairs)
        if len(out_shapes) == 1:
            oshp = out_shapes.pop()
            return "variable_in_fixed_out", f"→{oshp[0]}x{oshp[1]}"
        return "variable", "varies"

# ── main ─────────────────────────────────────────────────────────────────────

def classify_task(task_num, task_data):
    train = task_data.get('train', [])
    test  = task_data.get('test',  [])
    arcgen = task_data.get('arc-gen', [])
    tr_p = train + test  # use train+test for detection
    
    if not tr_p:
        return "empty", "no_data"
    
    # Get size info
    sz_type, sz_detail = size_relation(tr_p)
    
    # Get color info
    in_colors = set()
    out_colors = set()
    for p in tr_p:
        in_colors.update(np.array(p['input']).flatten().tolist())
        out_colors.update(np.array(p['output']).flatten().tolist())
    
    # Run detectors in order
    if is_identity(tr_p):
        return "identity", sz_detail
    
    r = is_rotation(tr_p)
    if r: return r, sz_detail
    
    r = is_flip(tr_p)
    if r: return r, sz_detail
    
    r = is_transpose(tr_p)
    if r: return r, sz_detail
    
    if is_color_remap(tr_p):
        return "color_remap", sz_detail
    
    r = is_scale(tr_p)
    if r: return r, sz_detail
    
    r = is_scale_up(tr_p)
    if r: return r, sz_detail
    
    r = is_tile(tr_p)
    if r: return r, sz_detail
    
    r = is_stack_v(tr_p)
    if r: return r, sz_detail
    
    r = is_stack_h(tr_p)
    if r: return r, sz_detail
    
    r = is_mirror_h(tr_p)
    if r: return r, sz_detail
    
    r = is_mirror_v(tr_p)
    if r: return r, sz_detail
    
    r = is_crop(tr_p)
    if r: return r, sz_detail
    
    r = is_roll(tr_p)
    if r: return r, sz_detail
    
    r = is_fractal(tr_p)
    if r: return r, sz_detail
    
    r = is_border(tr_p)
    if r: return r, sz_detail
    
    r = is_repeat_rows(tr_p)
    if r: return r, sz_detail
    
    r = is_repeat_cols(tr_p)
    if r: return r, sz_detail
    
    r = is_single_color_output(tr_p)
    if r: return r, sz_detail
    
    r = is_const_output(tr_p)
    if r: return r, sz_detail
    
    r = is_color_filter(tr_p)
    if r: return r, sz_detail
    
    r = is_geom_plus_remap(tr_p)
    if r: return r, sz_detail
    
    # Fallback: classify by size relationship
    return f"complex_{sz_type}", sz_detail


def main():
    rows = []
    category_counts = Counter()
    
    for task_num in range(1, 401):
        path = TASK_DIR / f"task{task_num:03d}.json"
        if not path.exists():
            continue
        task_data = load(path)
        
        train = task_data.get('train', [])
        test  = task_data.get('test', [])
        arcgen = task_data.get('arc-gen', [])
        tr_p = train + test
        
        # Basic stats
        n_train = len(train)
        n_test  = len(test)
        n_arcgen = len(arcgen)
        
        # First-example sizes
        if tr_p:
            ig0 = np.array(tr_p[0]['input'])
            og0 = np.array(tr_p[0]['output'])
            in_shape = f"{ig0.shape[0]}x{ig0.shape[1]}"
            out_shape = f"{og0.shape[0]}x{og0.shape[1]}"
            in_colors = sorted(set(ig0.flatten().tolist()))
            out_colors = sorted(set(og0.flatten().tolist()))
            same_size = (ig0.shape == og0.shape)
        else:
            in_shape = out_shape = "?"
            in_colors = out_colors = []
            same_size = False
        
        # Variable sizes?
        all_in_shapes = set()
        all_out_shapes = set()
        for p in tr_p:
            all_in_shapes.add(tuple(np.array(p['input']).shape))
            all_out_shapes.add(tuple(np.array(p['output']).shape))
        variable_in = len(all_in_shapes) > 1
        variable_out = len(all_out_shapes) > 1
        
        # Classify
        try:
            category, sz_detail = classify_task(task_num, task_data)
        except Exception as e:
            category = f"error:{type(e).__name__}"
            sz_detail = "?"
        
        category_counts[category] += 1
        
        rows.append({
            'task': f"task{task_num:03d}",
            'n_train': n_train,
            'n_test': n_test,
            'n_arcgen': n_arcgen,
            'in_shape': in_shape,
            'out_shape': out_shape,
            'same_size': same_size,
            'variable_in': variable_in,
            'variable_out': variable_out,
            'in_colors': str(in_colors),
            'out_colors': str(out_colors),
            'category': category,
            'size_detail': sz_detail,
        })
    
    # ── Write CSV ────────────────────────────────────────────────────────────
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'task', 'category', 'in_shape', 'out_shape', 'same_size',
            'variable_in', 'variable_out', 'in_colors', 'out_colors',
            'n_train', 'n_test', 'n_arcgen', 'size_detail'
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    
    # ── Print full table ─────────────────────────────────────────────────────
    print(f"{'TASK':<10} {'CATEGORY':<28} {'IN_SHAPE':<10} {'OUT_SHAPE':<10} {'SAME?':<6} {'VAR_IN':<7} {'VAR_OUT':<8} {'IN_COLORS':<25} {'OUT_COLORS':<25} {'TR':>3} {'TE':>3} {'AG':>4}")
    print("=" * 170)
    for r in rows:
        print(f"{r['task']:<10} {r['category']:<28} {r['in_shape']:<10} {r['out_shape']:<10} {'Y' if r['same_size'] else 'N':<6} {'Y' if r['variable_in'] else 'N':<7} {'Y' if r['variable_out'] else 'N':<8} {r['in_colors']:<25} {r['out_colors']:<25} {r['n_train']:>3} {r['n_test']:>3} {r['n_arcgen']:>4}")
    
    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("CATEGORY SUMMARY (400 tasks)")
    print("=" * 80)
    
    # Group into major families
    families = defaultdict(list)
    for cat, cnt in category_counts.most_common():
        if cat.startswith("complex_"):
            families["COMPLEX (needs ML)"].append((cat, cnt))
        elif cat in ("identity",):
            families["IDENTITY (zero-param)"].append((cat, cnt))
        elif "remap" in cat:
            families["COLOR REMAP (1x1 conv)"].append((cat, cnt))
        elif any(cat.startswith(x) for x in ("rot", "hflip", "vflip", "hvflip", "transpose", "antitranspose")):
            families["GEOMETRIC (gather)"].append((cat, cnt))
        elif any(cat.startswith(x) for x in ("scale", "tile", "stack", "mirror", "repeat", "fractal", "border")):
            families["SIZE/TILING (gather+mask)"].append((cat, cnt))
        elif any(cat.startswith(x) for x in ("crop",)):
            families["CROP (gather+mask)"].append((cat, cnt))
        elif any(cat.startswith(x) for x in ("const", "single_color")):
            families["CONSTANT OUTPUT"].append((cat, cnt))
        elif "color_filter" in cat:
            families["COLOR FILTER (1x1 conv)"].append((cat, cnt))
        elif "roll" in cat:
            families["ROLL/SHIFT (gather)"].append((cat, cnt))
        else:
            families["OTHER"].append((cat, cnt))
    
    total_handcrafted = 0
    total_ml = 0
    
    for family_name in [
        "IDENTITY (zero-param)", "GEOMETRIC (gather)", "COLOR REMAP (1x1 conv)",
        "SIZE/TILING (gather+mask)", "CROP (gather+mask)", "ROLL/SHIFT (gather)",
        "CONSTANT OUTPUT", "COLOR FILTER (1x1 conv)", "OTHER",
        "COMPLEX (needs ML)"
    ]:
        items = families.get(family_name, [])
        if not items:
            continue
        sub_total = sum(c for _, c in items)
        if family_name == "COMPLEX (needs ML)":
            total_ml += sub_total
        else:
            total_handcrafted += sub_total
        
        print(f"\n  {family_name} — {sub_total} tasks")
        for cat, cnt in sorted(items, key=lambda x: -x[1]):
            print(f"    {cat:<35} {cnt:>4}")
    
    print(f"\n{'='*80}")
    print(f"  HANDCRAFTED solvable (detectors):  {total_handcrafted}")
    print(f"  COMPLEX (needs ML/CNN/MLP):         {total_ml}")
    print(f"  TOTAL:                              {total_handcrafted + total_ml}")
    print(f"{'='*80}")
    
    # Size distribution for complex tasks
    complex_sizes = Counter()
    for r in rows:
        if r['category'].startswith("complex_"):
            complex_sizes[r['category'].replace("complex_", "")] += 1
    
    print(f"\n  COMPLEX task breakdown by size relationship:")
    for sz, cnt in complex_sizes.most_common():
        print(f"    {sz:<30} {cnt:>4}")
    
    print(f"\n  CSV written to: {OUT_CSV}")


if __name__ == "__main__":
    main()
