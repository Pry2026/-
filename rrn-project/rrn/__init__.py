"""
RRN-H: Ring Residual Network with Hedging
==========================================

A neural architecture that replaces the linear chain of standard ResNets
with a circular topology. Each block receives information from its ring
predecessor AND a diametric "hedge" signal, creating self-correcting
dynamics with equal gradient paths for every layer.

Quick start:
    from rrn import RingResidualNetwork

    model = RingResidualNetwork(
        input_dim=784,      # e.g. flattened MNIST
        output_dim=10,      # number of classes
        ring_dim=64,        # working dimension inside the ring
        num_blocks=4,       # N: blocks in the ring
        num_cycles=3,       # K: information circulation rounds
    )
    logits = model(x)       # x: (batch, 784)
"""

from rrn.core import RingResidualNetwork
from rrn.utils import compute_consensus, gradient_health

__all__ = ["RingResidualNetwork", "compute_consensus", "gradient_health"]
__version__ = "0.1.0"
