"""
Training utilities for RRN-H.

Provides a minimal but flexible training loop for classification tasks.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    flatten_input: bool = True,
) -> tuple[float, float]:
    """Run one training epoch.

    Returns (average_loss, accuracy).
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if flatten_input and x.dim() > 2:
            x = x.view(x.size(0), -1)

        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    flatten_input: bool = True,
) -> tuple[float, float]:
    """Evaluate model on a dataset.

    Returns (average_loss, accuracy).
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if flatten_input and x.dim() > 2:
            x = x.view(x.size(0), -1)
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    flatten_input: bool = True,
    verbose: bool = True,
) -> list[dict]:
    """Full training loop.

    Returns list of per-epoch metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = []
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, flatten_input)
        test_loss, test_acc = evaluate(model, test_loader, device, flatten_input)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        })

        if verbose:
            print(f"Epoch {epoch:>3d}  "
                  f"train_loss: {train_loss:.4f}  train_acc: {train_acc:.2%}  "
                  f"test_loss: {test_loss:.4f}  test_acc: {test_acc:.2%}")

    return history
