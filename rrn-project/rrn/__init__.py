"""
RRN-H: Ring Residual Network with Hedging — v2
================================================

A neural architecture replacing standard ResNet linear chains with a
circular topology and self-correcting "hedging" dynamics.

New in v2:
  - Cross-block attention mechanism
  - Adaptive residual connections (learnable α per block)
  - Multiple activation functions (GELU/ReLU/SiLU/Mish)
  - Conv stem for image datasets (CIFAR-10)
  - FashionMNIST & CIFAR-10 data loaders with augmentation
  - Cosine annealing LR, early stopping, gradient clipping
  - Training curves, confusion matrix, per-cycle accuracy
  - Full CLI with argparse

Quick start:
    from rrn import RingResidualNetwork
    model = RingResidualNetwork(784, 10, ring_dim=64, num_blocks=4, num_cycles=3)
    logits = model(x)

CLI training:
    python -m experiments.train_cli --dataset cifar10 --epochs 50

API:
    from rrn.data import get_dataloaders
    from rrn.train import fit
    from rrn.viz import plot_history_text, print_confusion_matrix
"""

from rrn.core import RingResidualNetwork
from rrn.utils import compute_consensus, gradient_health

__all__ = [
    "RingResidualNetwork",
    "compute_consensus",
    "gradient_health",
]
__version__ = "0.2.0"
