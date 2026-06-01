#!/usr/bin/env python3
"""
RRN-H Training CLI — single command to train on any dataset with any config.

Usage:
    python -m experiments.train_cli --dataset cifar10 --epochs 50
    python -m experiments.train_cli --dataset fashionmnist --ring-dim 128 --num-blocks 6
    python -m experiments.train_cli --dataset mnist --activation mish --use-attention

Full options: see --help
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rrn import RingResidualNetwork
from rrn.data import get_dataloaders
from rrn.train import fit, load_checkpoint
from rrn.viz import (
    compute_confusion_matrix,
    compute_per_cycle_accuracy,
    plot_history_text,
    print_confusion_matrix,
)


def build_model(args: argparse.Namespace, input_shape: tuple, num_classes: int) -> RingResidualNetwork:
    """Build RRN-H model from CLI args."""
    is_image = len(input_shape) == 3  # (C, H, W)

    return RingResidualNetwork(
        input_dim=input_shape[0] if is_image else input_shape[0],
        output_dim=num_classes,
        ring_dim=args.ring_dim,
        num_blocks=args.num_blocks,
        num_cycles=args.num_cycles,
        expansion=args.expansion,
        dropout=args.dropout,
        damping=args.damping,
        activation=args.activation,
        use_attention=args.use_attention,
        attn_heads=args.attn_heads,
        adaptive_residual=not args.no_adaptive_residual,
        use_conv_stem=is_image,
        img_size=input_shape[1] if is_image else 28,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RRN-H: Ring Residual Network Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m experiments.train_cli --dataset mnist
  python -m experiments.train_cli --dataset cifar10 --epochs 50 --ring-dim 128
  python -m experiments.train_cli --dataset fashionmnist --activation mish --use-attention""",
    )

    # ── Dataset ──────────────────────────────────────────────────────────
    parser.add_argument("--dataset", default="mnist",
                        choices=["mnist", "fashionmnist", "cifar10"],
                        help="Dataset to train on")
    parser.add_argument("--data-dir", default="./data",
                        help="Directory to store/download datasets")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable data augmentation")

    # ── Model architecture ──────────────────────────────────────────────
    parser.add_argument("--ring-dim", type=int, default=128,
                        help="Ring working dimension (D)")
    parser.add_argument("--num-blocks", type=int, default=6,
                        help="Number of blocks (N)")
    parser.add_argument("--num-cycles", type=int, default=3,
                        help="Number of cycles (K)")
    parser.add_argument("--expansion", type=int, default=2,
                        help="Hidden dim expansion factor")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--damping", type=float, default=0.5,
                        help="Jacobi damping γ")
    parser.add_argument("--activation", default="gelu",
                        choices=["gelu", "relu", "silu", "mish", "leaky_relu"],
                        help="Activation function")
    parser.add_argument("--use-attention", action="store_true",
                        help="Enable cross-block attention")
    parser.add_argument("--attn-heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--no-adaptive-residual", action="store_true",
                        help="Disable adaptive residual (use fixed)")

    # ── Training ────────────────────────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=50,
                        help="Max training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--scheduler", default="cosine",
                        choices=["cosine", "plateau", "none"],
                        help="LR scheduler type")
    parser.add_argument("--early-stop", type=int, default=15,
                        help="Early stopping patience")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping norm")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # ── Output ──────────────────────────────────────────────────────────
    parser.add_argument("--checkpoint-dir", default="./checkpoints",
                        help="Directory for model checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-epoch output")

    args = parser.parse_args()

    # ── Setup ───────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    log = logging.getLogger("rrn")

    log.info(f"Device: {device}")
    log.info(f"Dataset: {args.dataset}")

    # ── Data ────────────────────────────────────────────────────────────
    train_loader, test_loader, input_shape, num_classes = get_dataloaders(
        name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        augment=not args.no_augment,
    )
    is_image = len(input_shape) == 3

    # ── Model ───────────────────────────────────────────────────────────
    model = build_model(args, input_shape, num_classes)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model: {n_params:,} parameters")
    log.info(f"  ring_dim={args.ring_dim}, N={args.num_blocks}, "
             f"K={args.num_cycles}, act={args.activation}")
    if args.use_attention:
        log.info(f"  Attention: {args.attn_heads} heads")

    if args.resume:
        log.info(f"Resuming from {args.resume}")
        load_checkpoint(model, args.resume, device=device)
    model = model.to(device)

    # ── Train ───────────────────────────────────────────────────────────
    log.info("Starting training...")
    result = fit(
        model, train_loader, test_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler_kind=args.scheduler,
        early_stop_patience=args.early_stop,
        grad_clip=args.grad_clip,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        is_image_model=is_image,
        verbose=not args.quiet,
    )

    log.info(f"Best test accuracy: {result['best_acc']:.2%} (epoch {result['best_epoch']})")
    log.info(f"Total time: {result['total_time_s']:.0f}s")

    # ── Analysis ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING CURVES")
    print("=" * 60)
    print(plot_history_text(result["history"]))

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    cm = compute_confusion_matrix(model, test_loader, device, num_classes, is_image)
    print(print_confusion_matrix(cm))

    print("\n" + "=" * 60)
    print("PER-CYCLE ACCURACY")
    print("=" * 60)
    cycle_accs = compute_per_cycle_accuracy(model, test_loader, device, is_image)
    for i, acc in enumerate(cycle_accs):
        label = f"Cycle {i}" if i > 0 else "Init  "
        bar = "█" * int(acc * 30)
        print(f"  {label}: {acc:.2%} {bar}")

    # Block analysis
    print(f"\nBlock importance: {model.get_block_importance().tolist()}")
    alphas = model.get_adaptive_alphas()
    if alphas is not None:
        print(f"Adaptive alphas:  {alphas.flatten().tolist()}")


if __name__ == "__main__":
    main()
