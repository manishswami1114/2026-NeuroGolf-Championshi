#!/usr/bin/env python3
"""
Comprehensive analysis of 400 ARC-AGI NeuroGolf tasks.
Analyzes transformation patterns to understand requirements for ONNX networks.
"""

import json
import numpy as np
from collections import defaultdict
from pathlib import Path
import os

# Task directory
TASK_DIR = "/Users/manishswami/developer/Nuero-golf championship/neurogolf-2026/"

def load_task(task_id):
    """Load a single task JSON file."""
    task_num = str(task_id).zfill(3)
    path = os.path.join(TASK_DIR, f"task{task_num}.json")
    with open(path, 'r') as f:
        return json.load(f)

def analyze_size_transformation(task):
    """Analyze size relationships in a task."""
    train_data = task.get('train', [])
    if not train_data:
        return None
    
    sizes = []
    for example in train_data:
        inp = np.array(example['input'])
        out = np.array(example['output'])
        sizes.append({
            'input_shape': inp.shape,
            'output_shape': out.shape,
            'same_size': inp.shape == out.shape
        })
    return sizes

def get_color_palette(grid):
    """Get unique colors in a grid."""
    return set(np.array(grid).flatten().tolist())

def analyze_colors(task):
    """Analyze color usage."""
    train_data = task.get('train', [])
    all_colors = set()
    for example in train_data:
        all_colors.update(get_color_palette(example['input']))
        all_colors.update(get_color_palette(example['output']))
    return all_colors

def is_color_remap(example):
    """Check if transformation is just color remapping (1x1 convolution solvable)."""
    inp = np.array(example['input'])
    out = np.array(example['output'])
    
    # Must be same size
    if inp.shape != out.shape:
        return False
    
    # Check if there's a consistent color mapping
    inp_flat = inp.flatten()
    out_flat = out.flatten()
    
    color_map = {}
    for i, j in zip(inp_flat, out_flat):
        if i in color_map:
            if color_map[i] != j:
                return False
        else:
            color_map[i] = j
    return True

def is_rotation(example):
    """Check if output is a rotation of input."""
    inp = np.array(example['input'])
    out = np.array(example['output'])
    
    if inp.shape != out.shape:
        return None
    
    # Check 90, 180, 270 rotations
    for k in range(1, 4):
        if np.array_equal(out, np.rot90(inp, k)):
            return f"{k*90}°"
    return None

def is_reflection(example):
    """Check if output is a reflection of input."""
    inp = np.array(example['input'])
    out = np.array(example['output'])
    
    if inp.shape != out.shape:
        return None
    
    # Horizontal flip
    if np.array_equal(out, np.fliplr(inp)):
        return "horizontal"
    # Vertical flip
    if np.array_equal(out, np.flipud(inp)):
        return "vertical"
    return None

def is_tiling(example):
    """Check if output is input tiled in a grid."""
    inp = np.array(example['input'])
    out = np.array(example['output'])
    
    if out.shape[0] % inp.shape[0] != 0 or out.shape[1] % inp.shape[1] != 0:
        return None
    
    tile_h = out.shape[0] // inp.shape[0]
    tile_w = out.shape[1] // inp.shape[1]
    
    if tile_h > 1 or tile_w > 1:
        tiled = np.tile(inp, (tile_h, tile_w))
        if np.array_equal(out, tiled):
            return f"{tile_h}x{tile_w}"
    return None

def analyze_transformation_type(example):
    """Categorize the transformation type."""
    categories = []
    
    # Check color remap
    if is_color_remap(example):
        categories.append("color_remap")
    
    # Check rotation
    rot = is_rotation(example)
    if rot:
        categories.append(f"rotation_{rot}")
    
    # Check reflection
    refl = is_reflection(example)
    if refl:
        categories.append(f"reflection_{refl}")
    
    # Check tiling
    tile = is_tiling(example)
    if tile:
        categories.append(f"tiling_{tile}")
    
    inp_shape = np.array(example['input']).shape
    out_shape = np.array(example['output']).shape
    
    if inp_shape == out_shape and not categories:
        categories.append("same_size_transformation")
    elif inp_shape != out_shape:
        if out_shape[0] > inp_shape[0] or out_shape[1] > inp_shape[1]:
            categories.append("size_expansion")
        else:
            categories.append("size_reduction")
    
    return categories if categories else ["unknown"]

def analyze_specific_task(task_num):
    """Detailed analysis of a specific task."""
    try:
        task = load_task(task_num)
    except:
        return None
    
    train_data = task.get('train', [])
    if not train_data:
        return None
    
    result = {
        'task_id': f"task{str(task_num).zfill(3)}",
        'num_examples': len(train_data),
        'examples': []
    }
    
    for idx, example in enumerate(train_data[:2]):  # First 2 examples for detail
        inp = np.array(example['input'])
        out = np.array(example['output'])
        
        categories = analyze_transformation_type(example)
        
        ex_info = {
            'index': idx,
            'input_shape': inp.shape,
            'output_shape': out.shape,
            'input_colors': sorted(list(get_color_palette(example['input']))),
            'output_colors': sorted(list(get_color_palette(example['output']))),
            'transformation_types': categories,
            'input_grid': example['input'],
            'output_grid': example['output']
        }
        result['examples'].append(ex_info)
    
    return result

def main():
    """Main analysis function."""
    print("=" * 80)
    print("NEUROGOLF TASK ANALYSIS - 400 ARC-AGI Tasks")
    print("=" * 80)
    
    # Statistics
    same_size_count = 0
    diff_size_count = 0
    color_remap_count = 0
    rotation_count = 0
    reflection_count = 0
    tiling_count = 0
    
    input_sizes = defaultdict(int)
    output_sizes = defaultdict(int)
    transformation_types = defaultdict(int)
    
    # Analyze all tasks
    print("\nAnalyzing all 400 tasks...")
    for task_id in range(1, 401):
        try:
            task = load_task(task_id)
            train_data = task.get('train', [])
            
            if not train_data:
                continue
            
            # Analyze size relationship (using first example)
            first_example = train_data[0]
            inp = np.array(first_example['input'])
            out = np.array(first_example['output'])
            
            input_sizes[inp.shape] += 1
            output_sizes[out.shape] += 1
            
            if inp.shape == out.shape:
                same_size_count += 1
            else:
                diff_size_count += 1
            
            # Analyze transformation patterns
            for example in train_data:
                types = analyze_transformation_type(example)
                for t in types:
                    transformation_types[t] += 1
                
                if is_color_remap(example):
                    color_remap_count += 1
                if is_rotation(example):
                    rotation_count += 1
                if is_reflection(example):
                    reflection_count += 1
                if is_tiling(example):
                    tiling_count += 1
        
        except Exception as e:
            print(f"Error processing task {task_id}: {e}")
    
    # Report findings
    print("\n" + "=" * 80)
    print("1. SIZE MATCHING ANALYSIS")
    print("=" * 80)
    print(f"Tasks with SAME input/output size:      {same_size_count}")
    print(f"Tasks with DIFFERENT input/output size: {diff_size_count}")
    print(f"Percentage same-size:                   {100*same_size_count/(same_size_count+diff_size_count):.1f}%")
    
    print("\n" + "=" * 80)
    print("2. TRANSFORMATION PATTERN ANALYSIS")
    print("=" * 80)
    print(f"Tasks with Color Remapping:              {color_remap_count} examples")
    print(f"Tasks with Rotation:                     {rotation_count} examples")
    print(f"Tasks with Reflection:                   {reflection_count} examples")
    print(f"Tasks with Tiling:                       {tiling_count} examples")
    
    print("\nTransformation Type Frequency:")
    sorted_types = sorted(transformation_types.items(), key=lambda x: x[1], reverse=True)
    for trans_type, count in sorted_types[:15]:
        print(f"  {trans_type:40s}: {count:4d} examples")
    
    print("\n" + "=" * 80)
    print("3. INPUT SIZE DISTRIBUTION")
    print("=" * 80)
    sorted_input_sizes = sorted(input_sizes.items(), key=lambda x: x[1], reverse=True)
    for size, count in sorted_input_sizes[:10]:
        print(f"  {str(size):20s}: {count:3d} tasks")
    
    print("\n" + "=" * 80)
    print("4. OUTPUT SIZE DISTRIBUTION")
    print("=" * 80)
    sorted_output_sizes = sorted(output_sizes.items(), key=lambda x: x[1], reverse=True)
    for size, count in sorted_output_sizes[:10]:
        print(f"  {str(size):20s}: {count:3d} tasks")
    
    # Analyze specific tasks
    print("\n" + "=" * 80)
    print("5. SPECIFIC TASK ANALYSIS")
    print("=" * 80)
    
    for task_num in [2, 5, 10, 20, 50]:
        print(f"\n--- TASK {str(task_num).zfill(3)} ---")
        analysis = analyze_specific_task(task_num)
        
        if not analysis:
            print("Task not found or no training data")
            continue
        
        print(f"Number of training examples: {analysis['num_examples']}")
        
        for ex in analysis['examples']:
            print(f"\nExample {ex['index']}:")
            print(f"  Input shape:           {ex['input_shape']}")
            print(f"  Output shape:          {ex['output_shape']}")
            print(f"  Input colors:          {ex['input_colors']}")
            print(f"  Output colors:         {ex['output_colors']}")
            print(f"  Transformation types:  {', '.join(ex['transformation_types'])}")
            
            # Display grids
            print(f"\n  Input grid ({ex['input_shape'][0]}x{ex['input_shape'][1]}):")
            inp = np.array(ex['input_grid'])
            for row in inp:
                print(f"    {row.tolist()}")
            
            print(f"\n  Output grid ({ex['output_shape'][0]}x{ex['output_shape'][1]}):")
            out = np.array(ex['output_grid'])
            for row in out:
                print(f"    {row.tolist()}")
    
    # Summary of 1x1 convolution solvable tasks
    print("\n" + "=" * 80)
    print("6. TASKS SOLVABLE WITH 1x1 CONVOLUTION (Color Remapping Only)")
    print("=" * 80)
    
    conv_1x1_tasks = []
    for task_id in range(1, 401):
        try:
            task = load_task(task_id)
            train_data = task.get('train', [])
            
            # Check if ALL examples are color remaps
            all_color_remap = True
            for example in train_data:
                if not is_color_remap(example):
                    all_color_remap = False
                    break
            
            if all_color_remap and train_data:
                conv_1x1_tasks.append(task_id)
        except:
            pass
    
    print(f"Tasks solvable with 1x1 convolution: {len(conv_1x1_tasks)}")
    print(f"Percentage: {100*len(conv_1x1_tasks)/400:.1f}%")
    if conv_1x1_tasks[:20]:
        print(f"Example tasks: {[f'task{str(t).zfill(3)}' for t in conv_1x1_tasks[:20]]}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
