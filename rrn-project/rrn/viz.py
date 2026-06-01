"""
Visualization utilities: training curves, confusion matrix, consensus plots.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_confusion_matrix(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
    is_image_model: bool = False,
) -> torch.Tensor:
    """Returns (num_classes, num_classes) confusion matrix."""
    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if not is_image_model:
            x = x.view(x.size(0), -1)
        preds = model(x).argmax(1)
        for t, p in zip(y.tolist(), preds.tolist()):
            cm[t, p] += 1

    return cm


@torch.no_grad()
def compute_per_cycle_accuracy(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_image_model: bool = False,
) -> list[float]:
    """Accuracy after each ring cycle. Returns list of length K+1."""
    model.eval()
    N, D = model.num_blocks, model.ring_dim

    # Get a representative batch
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    if not is_image_model:
        x = x.view(x.size(0), -1)

    # Forward with all cycles
    _, all_cycles = model(x, return_all_cycles=True)

    accs = []
    for states in all_cycles:
        w = F.softmax(model.block_weights, dim=0)
        aggregated = (states * w.view(N, 1, 1)).sum(dim=0)
        logits = model.output_proj(aggregated)
        acc = (logits.argmax(1) == y).float().mean().item()
        accs.append(acc)

    return accs


def print_confusion_matrix(cm: torch.Tensor, class_names: list[str] | None = None) -> str:
    """Pretty-print a confusion matrix."""
    n = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n)]

    lines = []
    header = " " * 8 + "".join(f"{cn:>6}" for cn in class_names)
    lines.append(header)
    lines.append("-" * len(header))

    for i, name in enumerate(class_names):
        row = f"{name:<8}" + "".join(f"{cm[i, j].item():>6}" for j in range(n))
        lines.append(row)

    # Accuracy per class
    lines.append("-" * len(header))
    for i, name in enumerate(class_names):
        total = cm[i].sum().item()
        correct = cm[i, i].item()
        acc = correct / total if total > 0 else 0.0
        lines.append(f"{name:<8} acc={acc:.2%} ({correct}/{total})")

    return "\n".join(lines)


def plot_history_text(history: list[dict], width: int = 50) -> str:
    """ASCII plot of training history."""
    if not history:
        return "(no history)"

    epochs = [h["epoch"] for h in history]
    train_accs = [h["train_acc"] for h in history]
    test_accs = [h["test_acc"] for h in history]

    min_acc = min(min(train_accs), min(test_accs))
    max_acc = max(max(train_accs), max(test_accs))
    span = max_acc - min_acc or 0.01

    lines = ["Epoch  Train    Test"]
    lines.append("-" * (8 + width * 2 + 4))

    for i in range(len(history)):
        t_bar = int((train_accs[i] - min_acc) / span * width)
        v_bar = int((test_accs[i] - min_acc) / span * width)
        lines.append(
            f"{epochs[i]:>5d}  "
            f"{'█' * t_bar}{' ' * (width - t_bar)} "
            f"{'█' * v_bar}{' ' * (width - v_bar)} "
            f"{train_accs[i]:.2%}/{test_accs[i]:.2%}"
        )

    return "\n".join(lines)
