"""
Analysis utilities for RRN-H.

Provides tools to measure:
  - Consensus quality: how tightly ring blocks agree
  - Gradient health: uniformity of gradient norms across parameters
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_consensus(states: torch.Tensor) -> float:
    """Mean pairwise cosine distance between block states.

    Lower values indicate stronger consensus (blocks have converged
    to similar representations).

    Parameters
    ----------
    states : (N, B, D)
        States of N blocks for a batch of size B.

    Returns
    -------
    float
        Cosine distance ∈ [0, 2].  0 = perfect agreement, 2 = orthogonal.
    """
    N = states.shape[0]
    norms = F.normalize(states, dim=-1)                # (N, B, D)
    sim = torch.einsum("ibd,jbd->ijb", norms, norms)   # (N, N, B)
    mask = ~torch.eye(N, dtype=torch.bool, device=states.device)
    return float((1.0 - sim[mask].mean()).item())


def gradient_health(model: torch.nn.Module) -> dict[str, float]:
    """Measure gradient norm statistics across all trainable parameters.

    Call AFTER loss.backward().

    Parameters
    ----------
    model : nn.Module
        Model with accumulated gradients.

    Returns
    -------
    dict with keys: mean, std, min, max, cv (coefficient of variation).
    """
    norms = []
    for p in model.parameters():
        if p.grad is not None and p.requires_grad:
            norms.append(p.grad.norm().item())

    if not norms:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "cv": 0.0}

    t = torch.tensor(norms)
    mean = float(t.mean().item())
    std = float(t.std().item())
    return {
        "mean": mean,
        "std": std,
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "cv": float((std / (mean + 1e-8))),
    }
