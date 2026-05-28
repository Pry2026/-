"""
RRN-H core: Ring Residual Network with Hedging.

Mathematical formulation
------------------------
h_i^(0)   = shared_proj(x) + seed_i                   (diverse initialization)
h_i^(t+1) = (1-γ)·h_i^(t) + γ·(prev + MLP_i([prev, hedge]))
           where prev  = h_{i-1}^{(t)}                 (ring predecessor)
                 hedge = h_{i+N/2}^{(t)} - prev        (diametric correction)
output    = output_proj( Σ_i softmax(w_i) · h_i^(K) )  (weighted aggregation)

Key properties:
  - Equal gradient paths: every block has N distinct gradient routes
  - Self-correcting: hedge term pulls divergent blocks toward consensus
  - Depth-independent: more cycles improve convergence, not just depth
"""

from __future__ import annotations

from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class RingResidualNetwork(nn.Module):
    """Ring Residual Network with Hedging (RRN-H).

    Replaces the linear chain of standard ResNets with a circular topology
    where each block receives a "hedge" signal from the diametrically
    opposite block. After K cycles, all block outputs are aggregated via
    learned weights, ensuring every layer contributes equally.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input features (e.g. 784 for MNIST).
    output_dim : int
        Number of output classes or regression targets.
    ring_dim : int, default=256
        Working dimension inside the ring (D).
    num_blocks : int, default=8
        Number of residual blocks in the ring (N). Must be ≥ 2.
    num_cycles : int, default=4
        Number of full circulation rounds (K).
    expansion : int, default=4
        Hidden-dimension multiplier inside each block's MLP.
    dropout : float, default=0.0
        Dropout rate applied inside each block.
    damping : float, default=0.5
        Jacobi damping factor γ ∈ (0, 1]. Lower = more conservative updates.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        ring_dim: int = 256,
        num_blocks: int = 8,
        num_cycles: int = 4,
        expansion: int = 4,
        dropout: float = 0.0,
        damping: float = 0.5,
    ) -> None:
        super().__init__()
        if num_blocks < 2:
            raise ValueError(f"num_blocks must be ≥ 2, got {num_blocks}")

        self.num_blocks = num_blocks
        self.num_cycles = num_cycles
        self.ring_dim = ring_dim
        self.damping = damping
        N, D = num_blocks, ring_dim

        # ── Input projection ──────────────────────────────────────────
        self.shared_proj = nn.Sequential(
            nn.Linear(input_dim, D * 2),
            nn.GELU(),
            nn.Linear(D * 2, D),
            nn.LayerNorm(D),
        )
        self.block_seeds = nn.Parameter(torch.randn(N, D) * 0.5)

        # ── Ring blocks: fused MLP on concatenated [prev, hedge_diff] ──
        hidden = D * expansion
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(D * 2),
                nn.Linear(D * 2, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, D),
            )
            for _ in range(N)
        ])

        # ── Output aggregation ─────────────────────────────────────────
        self.block_weights = nn.Parameter(torch.ones(N) / N)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D * 2),
            nn.GELU(),
            nn.Linear(D * 2, output_dim),
        )

        # ── Precomputed ring indices ───────────────────────────────────
        self.register_buffer("_prev_idx", torch.tensor(
            [(i - 1) % N for i in range(N)], dtype=torch.long
        ))
        self.register_buffer("_opp_idx", torch.tensor(
            [(i + N // 2) % N for i in range(N)], dtype=torch.long
        ))

    def _ring_step(self, states: torch.Tensor) -> torch.Tensor:
        """Single damped-Jacobi ring update.

        Parameters
        ----------
        states : (N, B, D)
            Current states of all N blocks for a batch of size B.

        Returns
        -------
        (N, B, D)
            Updated states after one parallel ring circulation.
        """
        N, B, D = states.shape

        prev = states[self._prev_idx]                # (N, B, D)
        opp = states[self._opp_idx]                  # (N, B, D)
        hedge = opp - prev                            # (N, B, D)
        fused = torch.cat([prev, hedge], dim=-1)      # (N, B, 2D)

        updates = torch.stack([
            self.blocks[i](fused[i]) for i in range(N)
        ])

        gamma = self.damping
        return (1 - gamma) * states + gamma * (prev + updates)

    def forward(
        self,
        x: torch.Tensor,
        return_all_cycles: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Forward pass.

        Parameters
        ----------
        x : (B, input_dim)
            Input batch.
        return_all_cycles : bool
            If True, also returns the block states after every cycle.

        Returns
        -------
        logits : (B, output_dim)
            Model predictions.
        cycles : list of (N, B, D) tensors, optional
            Block states at each cycle (including initialization at index 0).
        """
        B = x.shape[0]
        N, D = self.num_blocks, self.ring_dim

        # Initialize diverse block states
        base = self.shared_proj(x)                          # (B, D)
        states = base.unsqueeze(0) + self.block_seeds.unsqueeze(1)  # (N, B, D)

        all_cycles: list[torch.Tensor] | None = [states] if return_all_cycles else None

        for _ in range(self.num_cycles):
            states = self._ring_step(states)
            if return_all_cycles:
                assert all_cycles is not None
                all_cycles.append(states)

        # Weighted aggregation across blocks
        w = F.softmax(self.block_weights, dim=0)          # (N,)
        aggregated = (states * w.view(N, 1, 1)).sum(dim=0)  # (B, D)
        out = self.output_proj(aggregated)                  # (B, output_dim)

        if return_all_cycles:
            return out, all_cycles
        return out

    def get_block_importance(self) -> torch.Tensor:
        """Softmax-normalized contribution weight for each block.

        Returns
        -------
        (num_blocks,) tensor of probabilities summing to 1.
        """
        return F.softmax(self.block_weights, dim=0)

    def extra_repr(self) -> str:
        return (
            f"ring_dim={self.ring_dim}, num_blocks={self.num_blocks}, "
            f"num_cycles={self.num_cycles}, damping={self.damping}"
        )
