"""NeuroGolf 2026 - Solve ARC-AGI tasks with minimal ONNX neural networks."""

import json
import math
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import onnxruntime
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHANNELS = 10
HEIGHT = WIDTH = 30
GRID_SHAPE = [1, CHANNELS, HEIGHT, WIDTH]
DATA_DIR = Path(__file__).parent
SUBMISSION_DIR = DATA_DIR / "submission"
SUBMISSION_DIR.mkdir(exist_ok=True)

IR_VERSION = 10
OPSET_IMPORTS = [onnx.helper.make_opsetid("", 10)]
DATA_TYPE = onnx.TensorProto.FLOAT

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_task(task_num: int) -> dict:
    with open(DATA_DIR / f"task{task_num:03d}.json") as f:
        return json.load(f)


def grid_to_tensor(grid: list) -> torch.Tensor:
    """Convert a 2D grid of color indices to a [1, 10, 30, 30] one-hot tensor."""
    t = torch.zeros(1, CHANNELS, HEIGHT, WIDTH, dtype=torch.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            t[0, color, r, c] = 1.0
    return t


def examples_to_tensors(examples: list):
    """Convert list of {input, output} dicts to paired tensor lists."""
    inputs, outputs = [], []
    for ex in examples:
        inputs.append(grid_to_tensor(ex["input"]))
        outputs.append(grid_to_tensor(ex["output"]))
    return torch.cat(inputs, dim=0), torch.cat(outputs, dim=0)


def check_accuracy(model, inputs, targets):
    """Check exact-match accuracy after thresholding at 0."""
    model.eval()
    with torch.no_grad():
        preds = model(inputs)
        binary_preds = (preds > 0.0).float()
        match = torch.all(binary_preds == targets, dim=(1, 2, 3))
        return match.float().mean().item(), match.sum().item(), len(match)


# ---------------------------------------------------------------------------
# Architecture Tiers (PyTorch)
# ---------------------------------------------------------------------------

class SingleConv(nn.Module):
    """Tier 1: Single conv layer, no bias."""
    def __init__(self, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(CHANNELS, CHANNELS, kernel_size, padding=pad, bias=False)

    def forward(self, x):
        return self.conv(x)


class BottleneckConv(nn.Module):
    """Tier 2: 1x1 down -> KxK -> 1x1 up, no bias."""
    def __init__(self, hidden=4, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(CHANNELS, hidden, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size, padding=pad, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden, CHANNELS, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class ResidualConv(nn.Module):
    """Tier 3: Conv blocks with residual connection."""
    def __init__(self, hidden=8, kernel_size=3, num_blocks=2):
        super().__init__()
        pad = kernel_size // 2
        layers = []
        layers.append(nn.Conv2d(CHANNELS, hidden, kernel_size, padding=pad, bias=False))
        layers.append(nn.ReLU())
        for _ in range(num_blocks - 1):
            layers.append(nn.Conv2d(hidden, hidden, kernel_size, padding=pad, bias=False))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(hidden, CHANNELS, kernel_size, padding=pad, bias=False))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.body(x)


class DilatedConv(nn.Module):
    """Tier 4: Dilated convolutions for large receptive field."""
    def __init__(self, hidden=8, kernel_size=3):
        super().__init__()
        pad1 = kernel_size // 2
        pad2 = (kernel_size // 2) * 2
        pad4 = (kernel_size // 2) * 4
        self.net = nn.Sequential(
            nn.Conv2d(CHANNELS, hidden, kernel_size, padding=pad1, dilation=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size, padding=pad2, dilation=2, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size, padding=pad4, dilation=4, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden, CHANNELS, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class MultiScaleConv(nn.Module):
    """Tier 5: Multi-scale with global context."""
    def __init__(self, hidden=8):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(CHANNELS, hidden, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.ReLU(),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Conv2d(CHANNELS, hidden, 1, bias=False),
            nn.ReLU(),
        )
        self.combine = nn.Conv2d(hidden * 2, CHANNELS, 1, bias=False)

    def forward(self, x):
        local_feat = self.local(x)
        global_feat = self.global_fc(self.global_pool(x))
        global_feat = global_feat.expand_as(local_feat)
        combined = torch.cat([local_feat, global_feat], dim=1)
        return self.combine(combined)


class DeepResidual(nn.Module):
    """Tier 5b: Deeper residual network with more capacity."""
    def __init__(self, hidden=16, kernel_size=3, num_blocks=3):
        super().__init__()
        pad = kernel_size // 2
        self.input_proj = nn.Conv2d(CHANNELS, hidden, 1, bias=False)
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResBlock(hidden, kernel_size))
        self.blocks = nn.Sequential(*blocks)
        self.output_proj = nn.Conv2d(hidden, CHANNELS, 1, bias=False)
        self.skip = nn.Identity()

    def forward(self, x):
        h = self.input_proj(x)
        h = self.blocks(h)
        return self.output_proj(h) + x


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=False),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.net(x))


# ---------------------------------------------------------------------------
# Architecture configurations to try (ordered by expected cost)
# ---------------------------------------------------------------------------

def get_architecture_configs():
    """Return list of (name, constructor) in order of increasing cost."""
    configs = [
        # Tier 1: Single conv
        ("conv1x1", lambda: SingleConv(kernel_size=1)),
        ("conv3x3", lambda: SingleConv(kernel_size=3)),
        ("conv5x5", lambda: SingleConv(kernel_size=5)),
        ("conv7x7", lambda: SingleConv(kernel_size=7)),
        # Tier 2: Bottleneck
        ("bneck_h4_k3", lambda: BottleneckConv(hidden=4, kernel_size=3)),
        ("bneck_h4_k5", lambda: BottleneckConv(hidden=4, kernel_size=5)),
        ("bneck_h8_k3", lambda: BottleneckConv(hidden=8, kernel_size=3)),
        ("bneck_h8_k5", lambda: BottleneckConv(hidden=8, kernel_size=5)),
        # Tier 3: Residual
        ("res_h8_b2", lambda: ResidualConv(hidden=8, num_blocks=2)),
        ("res_h16_b2", lambda: ResidualConv(hidden=16, num_blocks=2)),
        ("res_h8_b3", lambda: ResidualConv(hidden=8, num_blocks=3)),
        # Tier 4: Dilated
        ("dilated_h8", lambda: DilatedConv(hidden=8)),
        ("dilated_h16", lambda: DilatedConv(hidden=16)),
        # Tier 5: Global + Deep
        ("multiscale_h8", lambda: MultiScaleConv(hidden=8)),
        ("multiscale_h16", lambda: MultiScaleConv(hidden=16)),
        ("deepres_h16_b3", lambda: DeepResidual(hidden=16, num_blocks=3)),
        ("deepres_h32_b3", lambda: DeepResidual(hidden=32, num_blocks=3)),
        ("deepres_h16_b5", lambda: DeepResidual(hidden=16, num_blocks=5)),
    ]
    return configs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def compute_loss(preds, targets):
    """Combined loss: BCE with logits (for correct sign) + MSE (for magnitude)."""
    # BCE with logits: treats preds as logits, targets as 0/1
    bce = nn.functional.binary_cross_entropy_with_logits(preds, targets)
    # MSE component for magnitude guidance
    mse = nn.functional.mse_loss(preds, targets * 2 - 1)  # target: -1 or +1
    return bce + 0.1 * mse


def train_model(model, train_inputs, train_targets, max_epochs=1500, lr=0.003,
                patience=300, verbose=False, batch_size=64, early_fail_epochs=150):
    """Train a model to match input->output pairs exactly.

    Returns (success, best_accuracy, epochs_trained, best_state_dict).
    early_fail_epochs: if accuracy is still 0% after this many epochs, abort early.
    """
    model = model.to(DEVICE)
    train_inputs = train_inputs.to(DEVICE)
    train_targets = train_targets.to(DEVICE)

    n_examples = train_inputs.shape[0]
    best_acc = 0.0
    best_correct = 0
    best_state = None
    epochs_no_improve = 0

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=400, T_mult=2)

    for epoch in range(max_epochs):
        model.train()

        # Full-batch or mini-batch
        if n_examples <= batch_size:
            optimizer.zero_grad()
            preds = model(train_inputs)
            loss = compute_loss(preds, train_targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        else:
            perm = torch.randperm(n_examples, device=DEVICE)
            for i in range(0, n_examples, batch_size):
                idx = perm[i:i+batch_size]
                optimizer.zero_grad()
                preds = model(train_inputs[idx])
                loss = compute_loss(preds, train_targets[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        scheduler.step()

        # Check accuracy every 25 epochs
        if (epoch + 1) % 25 == 0:
            acc, correct, total = check_accuracy(model, train_inputs, train_targets)
            if verbose and (epoch + 1) % 200 == 0:
                print(f"  Epoch {epoch+1}: loss={loss.item():.6f}, "
                      f"acc={correct}/{total} ({acc:.1%})")

            # Early fail: if no progress at all after early_fail_epochs
            if (epoch + 1) >= early_fail_epochs and best_correct == 0:
                return False, 0.0, epoch + 1, None

            if correct > best_correct:
                best_correct = correct
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 25

            if acc == 1.0:
                return True, acc, epoch + 1, best_state

            if epochs_no_improve >= patience:
                break

    # Restore best state
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(DEVICE)

    return best_acc == 1.0, best_acc, epoch + 1, best_state


# ---------------------------------------------------------------------------
# ONNX Export
# ---------------------------------------------------------------------------

def export_to_onnx(model, task_num: int) -> str:
    """Export PyTorch model to ONNX format."""
    model = model.cpu().eval()
    filepath = str(SUBMISSION_DIR / f"task{task_num:03d}.onnx")
    dummy = torch.randn(1, CHANNELS, HEIGHT, WIDTH)

    export_kwargs = dict(
        opset_version=10,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
    )
    import inspect
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False
    torch.onnx.export(model, dummy, filepath, **export_kwargs)
    return filepath


# ---------------------------------------------------------------------------
# Validation using competition utilities
# ---------------------------------------------------------------------------

def validate_onnx(filepath: str, examples: dict) -> tuple:
    """Validate ONNX model against all examples. Returns (all_pass, score, details)."""
    # Check file exists and size
    fpath = Path(filepath)
    if not fpath.exists():
        return False, 0.0, "File not found"
    if fpath.stat().st_size > 1.44 * 1024 * 1024:
        return False, 0.0, f"File too large: {fpath.stat().st_size} bytes"

    try:
        session = onnxruntime.InferenceSession(filepath)
    except Exception as e:
        return False, 0.0, f"Failed to load: {e}"

    # Test all examples
    all_examples = examples.get("train", []) + examples.get("test", []) + examples.get("arc-gen", [])
    correct = 0
    total = len(all_examples)

    for ex in all_examples:
        inp = grid_to_numpy(ex["input"])
        expected = grid_to_numpy(ex["output"])
        try:
            result = session.run(["output"], {"input": inp})
            pred = (result[0] > 0.0).astype(np.float32)
            if np.array_equal(pred, expected):
                correct += 1
        except Exception:
            pass

    if correct < total:
        return False, 0.0, f"Accuracy: {correct}/{total}"

    # Score the network
    try:
        import onnx_tool
        model_obj = onnx_tool.loadmodel(filepath, {'verbose': False})
        g = model_obj.graph
        g.graph_reorder_nodes()
        g.shape_infer(None)
        g.profile()
        if not g.valid_profile:
            return False, 0.0, "Invalid profile"
        macs = int(sum(g.macs))
        memory = int(g.memory)
        params = int(g.params)
        cost = macs + memory + params
        score = max(1.0, 25.0 - math.log(cost))
        return True, score, f"MACs={macs}, mem={memory}, params={params}, score={score:.3f}"
    except Exception as e:
        return True, 1.0, f"Passed but scoring failed: {e}"


def grid_to_numpy(grid: list) -> np.ndarray:
    """Convert grid to numpy array matching neurogolf_utils.convert_to_numpy."""
    t = np.zeros((1, CHANNELS, HEIGHT, WIDTH), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            t[0, color, r, c] = 1.0
    return t


# ---------------------------------------------------------------------------
# Handcrafted solutions
# ---------------------------------------------------------------------------

def solve_color_remap(task_num: int, examples: dict) -> str | None:
    """Try to solve as a pure color remap (1x1 conv). Returns filepath or None."""
    all_examples = examples["train"] + examples["test"]

    # Build color map
    color_map = {}
    for ex in all_examples:
        inp, out = ex["input"], ex["output"]
        if len(inp) != len(out) or len(inp[0]) != len(out[0]):
            return None
        for r in range(len(inp)):
            for c in range(len(inp[0])):
                ic, oc = inp[r][c], out[r][c]
                if ic in color_map:
                    if color_map[ic] != oc:
                        return None
                else:
                    color_map[ic] = oc

    # Build 1x1 conv weights: W[out_ch, in_ch, 1, 1]
    # For color i -> color j: W[j, i] = 1.0, W[other, i] = 0.0
    weights = np.zeros((CHANNELS, CHANNELS, 1, 1), dtype=np.float32)
    for in_color, out_color in color_map.items():
        weights[out_color, in_color, 0, 0] = 1.0
    # For unmapped colors, keep identity
    for c in range(CHANNELS):
        if c not in color_map:
            weights[c, c, 0, 0] = 1.0

    # Build ONNX model
    x = onnx.helper.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = onnx.helper.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    w = onnx.numpy_helper.from_array(weights, name="W")
    node = onnx.helper.make_node("Conv", ["input", "W"], ["output"],
                                  kernel_shape=[1, 1], pads=[0, 0, 0, 0])
    graph = onnx.helper.make_graph([node], "graph", [x], [y], [w])
    model = onnx.helper.make_model(graph, ir_version=IR_VERSION,
                                    opset_imports=OPSET_IMPORTS)

    filepath = str(SUBMISSION_DIR / f"task{task_num:03d}.onnx")
    onnx.save(model, filepath)

    passed, score, details = validate_onnx(filepath, examples)
    if passed:
        return filepath
    else:
        os.remove(filepath)
        return None


# ---------------------------------------------------------------------------
# Main solve pipeline
# ---------------------------------------------------------------------------

def solve_task(task_num: int, verbose=False, data_dir=None) -> dict:
    """Attempt to solve a single task. Returns result dict."""
    examples = load_task(task_num) if data_dir is None else load_task_from(task_num, data_dir)
    result = {"task": task_num, "solved": False, "score": 0.0, "tier": None, "details": ""}

    # Gather all examples for training
    all_examples = examples["train"] + examples["test"] + examples.get("arc-gen", [])
    train_inputs, train_targets = examples_to_tensors(all_examples)
    n_examples = len(all_examples)

    # --- Try handcrafted color remap first ---
    filepath = solve_color_remap(task_num, examples)
    if filepath:
        _, score, details = validate_onnx(filepath, examples)
        result.update(solved=True, score=score, tier="handcrafted", details=details)
        if verbose:
            print(f"  Task {task_num:3d}: SOLVED (handcrafted) - {details}")
        return result

    # --- Try learned architectures with escalating complexity ---
    configs = get_architecture_configs()

    for arch_name, arch_fn in configs:
        model = arch_fn()
        param_count = sum(p.numel() for p in model.parameters())

        # Scaled training budget
        max_epochs = 2000 if param_count < 1000 else (1500 if param_count < 10000 else 1000)
        bs = min(64, n_examples)

        success, best_acc, epochs, best_state = train_model(
            model, train_inputs, train_targets,
            max_epochs=max_epochs, lr=0.003, patience=300,
            verbose=verbose, batch_size=bs, early_fail_epochs=150
        )

        if success:
            if best_state:
                model.load_state_dict(best_state)
            filepath = export_to_onnx(model, task_num)
            passed, score, details = validate_onnx(filepath, examples)

            if passed:
                result.update(solved=True, score=score, tier=arch_name, details=details)
                if verbose:
                    print(f"  Task {task_num:3d}: SOLVED ({arch_name}) - {details}")
                return result
            else:
                # Training matched but ONNX validation didn't - retry with lr=0.01
                os.remove(filepath)
                model2 = arch_fn()
                s2, a2, e2, st2 = train_model(
                    model2, train_inputs, train_targets,
                    max_epochs=max_epochs, lr=0.01, patience=300,
                    verbose=False, batch_size=bs, early_fail_epochs=150
                )
                if s2 and st2:
                    model2.load_state_dict(st2)
                    filepath = export_to_onnx(model2, task_num)
                    passed, score, details = validate_onnx(filepath, examples)
                    if passed:
                        result.update(solved=True, score=score, tier=arch_name, details=details)
                        if verbose:
                            print(f"  Task {task_num:3d}: SOLVED ({arch_name}, retry) - {details}")
                        return result
                    os.remove(filepath)
        elif verbose and best_acc > 0:
            print(f"  Task {task_num:3d}: {arch_name} partial (acc={best_acc:.1%}, {epochs} ep)")

    if verbose:
        print(f"  Task {task_num:3d}: UNSOLVED")
    return result


def load_task_from(task_num: int, data_dir: str) -> dict:
    """Load task from a specific directory."""
    with open(Path(data_dir) / f"task{task_num:03d}.json") as f:
        return json.load(f)


def solve_all(task_range=None, verbose=True):
    """Solve all tasks and create submission."""
    if task_range is None:
        task_range = range(1, 401)

    results = []
    total_score = 0.0
    solved_count = 0
    start_time = time.time()

    for task_num in task_range:
        t0 = time.time()
        result = solve_task(task_num, verbose=verbose)
        elapsed = time.time() - t0

        results.append(result)
        if result["solved"]:
            solved_count += 1
            total_score += result["score"]

        if verbose:
            print(f"  [{elapsed:.1f}s] Task {task_num:3d}: "
                  f"{'SOLVED' if result['solved'] else 'FAILED'} "
                  f"(cumulative: {solved_count} solved, {total_score:.1f} points)")

    elapsed_total = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Results: {solved_count}/{len(list(task_range))} tasks solved")
    print(f"Total score: {total_score:.1f}")
    print(f"Total time: {elapsed_total:.0f}s")
    print(f"{'='*60}")

    # Create submission zip
    create_submission_zip()

    # Save results log
    results_path = DATA_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    return results


def create_submission_zip():
    """Create submission.zip from all .onnx files in submission/."""
    zip_path = DATA_DIR / "submission.zip"
    onnx_files = sorted(SUBMISSION_DIR.glob("task*.onnx"))
    if not onnx_files:
        print("No ONNX files to zip!")
        return

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in onnx_files:
            zf.write(f, f.name)

    print(f"Created {zip_path} with {len(onnx_files)} files "
          f"({zip_path.stat().st_size / 1024:.0f} KB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Solve specific tasks: python solve.py 1 2 3 or python solve.py 1-10
        tasks = []
        for arg in sys.argv[1:]:
            if "-" in arg:
                start, end = arg.split("-")
                tasks.extend(range(int(start), int(end) + 1))
            else:
                tasks.append(int(arg))
        solve_all(task_range=tasks, verbose=True)
    else:
        solve_all(verbose=True)
