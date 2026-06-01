"""
Training utilities v2: LR schedules, early stopping, gradient clipping, checkpointing.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader


# ─── LR Scheduler factory ────────────────────────────────────────────────────

def create_scheduler(
    optimizer: torch.optim.Optimizer,
    kind: str = "cosine",
    T_max: int = 100,
    factor: float = 0.5,
    patience: int = 10,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Create learning-rate scheduler.

    kind: "cosine" | "plateau" | "none"
    """
    if kind == "cosine":
        return CosineAnnealingLR(optimizer, T_max=T_max)
    elif kind == "plateau":
        return ReduceLROnPlateau(optimizer, mode="max", factor=factor,
                                 patience=patience)
    return None


# ─── Early Stopping ──────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when validation metric stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.best_epoch = 0

    def step(self, metric: float, epoch: int) -> bool:
        """Returns True if should stop."""
        improved = (
            metric > self.best + self.min_delta
            if self.mode == "max"
            else metric < self.best - self.min_delta
        )
        if improved:
            self.best = metric
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
        return self.counter >= self.patience


# ─── Model save/load ─────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    metrics: dict | None = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict = {
        "model_state": model.state_dict(),
        "epoch": epoch,
    }
    if optimizer:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    if metrics:
        checkpoint["metrics"] = metrics
    torch.save(checkpoint, path)


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> dict:
    checkpoint = torch.load(path, map_location=device or "cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint


# ─── Train / Eval loops ──────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 0.0,
    is_image_model: bool = False,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if not is_image_model:
            x = x.view(x.size(0), -1)

        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item() * x.size(0)
        # Re-forward for accuracy (cheap on small models)
        with torch.no_grad():
            if not is_image_model:
                xf = x.view(x.size(0), -1)
            else:
                xf = x
            correct += (model(xf).argmax(1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_image_model: bool = False,
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if not is_image_model:
            x = x.view(x.size(0), -1)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


# ─── Full training loop ──────────────────────────────────────────────────────

def fit(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    scheduler_kind: str = "cosine",
    early_stop_patience: int = 15,
    grad_clip: float = 1.0,
    device: torch.device | None = None,
    checkpoint_dir: str | None = None,
    is_image_model: bool = False,
    verbose: bool = True,
) -> dict:
    """Full training loop with all bells and whistles.

    Returns dict with keys: history, best_acc, best_epoch, total_time_s.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = create_scheduler(optimizer, scheduler_kind, T_max=epochs)
    early_stop = EarlyStopping(patience=early_stop_patience, mode="max")

    history: list[dict] = []
    best_acc = 0.0
    best_epoch = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, grad_clip, is_image_model
        )
        test_loss, test_acc = evaluate(model, test_loader, device, is_image_model)

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(test_acc)
            else:
                scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "test_loss": round(test_loss, 6),
            "test_acc": round(test_acc, 6),
            "lr": optimizer.param_groups[0]["lr"],
        })

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            if checkpoint_dir:
                save_checkpoint(model, f"{checkpoint_dir}/best.pt",
                                optimizer, epoch, {"test_acc": test_acc})

        if verbose:
            lr_str = f"{optimizer.param_groups[0]['lr']:.2e}"
            print(f"Epoch {epoch:>3d}  "
                  f"train_loss: {train_loss:.4f}  train_acc: {train_acc:.2%}  "
                  f"test_loss: {test_loss:.4f}  test_acc: {test_acc:.2%}  "
                  f"lr: {lr_str}")

        if early_stop.step(test_acc, epoch):
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break

    elapsed = time.time() - t0
    if checkpoint_dir:
        save_checkpoint(model, f"{checkpoint_dir}/last.pt", optimizer, epoch)

    # Save history
    if checkpoint_dir:
        with open(f"{checkpoint_dir}/history.json", "w") as f:
            json.dump(history, f, indent=2)

    return {
        "history": history,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "total_time_s": elapsed,
    }
