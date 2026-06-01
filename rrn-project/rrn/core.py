"""
RRN-H v2 core: Ring Residual Network with Hedging.

New in v2:
  - Cross-block attention (optional, per-cycle)
  - Adaptive residual: learned α_i per block
  - Multiple activation choices: gelu, relu, silu, mish
  - Convolutional stem for image inputs (CIFAR-10)

Mathematical core (unchanged):
  h_i^(t+1) = (1-γ)·h_i^(t) + γ·(prev + α_i · MLP_i([prev, hedge]))
  hedge = h_{opposite} - prev
"""

from __future__ import annotations

from typing import List, Literal, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Activations ─────────────────────────────────────────────────────────────

def _get_activation(name: str) -> nn.Module:
    """Resolve activation by name."""
    table: dict[str, type[nn.Module]] = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "mish": nn.Mish,
        "leaky_relu": nn.LeakyReLU,
    }
    if name not in table:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(table)}.")
    cls = table[name]
    return cls() if name != "leaky_relu" else cls(0.1)


# ─── Attention ───────────────────────────────────────────────────────────────

class BlockAttention(nn.Module):
    """Cross-block self-attention: blocks attend to each other."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=False
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """states: (N, B, D) → (N, B, D) with residual."""
        s_norm = self.norm(states)
        attn_out, _ = self.attn(s_norm, s_norm, s_norm)
        return states + attn_out


# ─── Adaptive Residual Block ─────────────────────────────────────────────────

class AdaptiveRingBlock(nn.Module):
    """Single ring block with learnable residual strength α."""

    def __init__(
        self,
        dim: int,
        expansion: int = 4,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        hidden = dim * expansion
        act = _get_activation(activation)

        self.net = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, hidden),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        # Learnable residual strength (init 1.0 = standard residual)
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, prev: torch.Tensor, hedge: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([prev, hedge], dim=-1)
        return prev + self.alpha * self.net(fused)


# ─── Convolutional Stem (for image inputs) ───────────────────────────────────

class ConvStem(nn.Module):
    """Lightweight conv stem to extract features from (C, H, W) images."""

    def __init__(self, in_channels: int, out_dim: int, img_size: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
        )
        # Compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            flat = self.conv(dummy).shape[1]
        self.proj = nn.Linear(flat, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(x))


# ─── Full Network ────────────────────────────────────────────────────────────

class RingResidualNetwork(nn.Module):
    """RRN-H v2: ring topology with hedging, attention, and adaptive residuals.

    Parameters
    ----------
    input_dim : int
        Feature dimension. If use_conv_stem=True, this is in_channels (1 or 3).
    output_dim : int
        Number of classes.
    ring_dim : int
        Internal ring working dimension.
    num_blocks : int
        Number of blocks in the ring (N ≥ 2).
    num_cycles : int
        Full circulation rounds (K).
    expansion : int
        Hidden-dim multiplier inside each block's MLP.
    dropout : float
        Dropout rate.
    damping : float
        Jacobi damping γ ∈ (0, 1].
    activation : str
        "gelu" | "relu" | "silu" | "mish" | "leaky_relu".
    use_attention : bool
        Add cross-block self-attention after each cycle.
    attn_heads : int
        Number of attention heads (only if use_attention=True).
    adaptive_residual : bool
        Use learnable α per block instead of fixed residual.
    use_conv_stem : bool
        Use conv stem for (C,H,W) image input (e.g. CIFAR-10).
    img_size : int
        Input image size (only if use_conv_stem=True).
    """

    def __init__(
        self,
        input_dim: int = 784,
        output_dim: int = 10,
        ring_dim: int = 256,
        num_blocks: int = 8,
        num_cycles: int = 4,
        expansion: int = 4,
        dropout: float = 0.0,
        damping: float = 0.5,
        activation: str = "gelu",
        use_attention: bool = False,
        attn_heads: int = 4,
        adaptive_residual: bool = True,
        use_conv_stem: bool = False,
        img_size: int = 32,
    ) -> None:
        super().__init__()
        if num_blocks < 2:
            raise ValueError(f"num_blocks must be ≥ 2, got {num_blocks}")

        self.num_blocks = num_blocks
        self.num_cycles = num_cycles
        self.ring_dim = ring_dim
        self.damping = damping
        self.use_conv_stem = use_conv_stem
        N, D = num_blocks, ring_dim

        # ── Input ────────────────────────────────────────────────────
        if use_conv_stem:
            self.conv_stem = ConvStem(input_dim, D, img_size)
            self.shared_proj = nn.Identity()
        else:
            self.conv_stem = None  # type: ignore[assignment]
            self.shared_proj = nn.Sequential(
                nn.Linear(input_dim, D * 2),
                _get_activation(activation),
                nn.Linear(D * 2, D),
                nn.LayerNorm(D),
            )
        self.block_seeds = nn.Parameter(torch.randn(N, D) * 0.5)

        # ── Ring blocks ──────────────────────────────────────────────
        if adaptive_residual:
            self.blocks = nn.ModuleList([
                AdaptiveRingBlock(D, expansion, dropout, activation)
                for _ in range(N)
            ])
        else:
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(D * 2),
                    nn.Linear(D * 2, D * expansion),
                    _get_activation(activation),
                    nn.Dropout(dropout),
                    nn.Linear(D * expansion, D),
                )
                for _ in range(N)
            ])

        # ── Attention (optional) ─────────────────────────────────────
        self.attention = (
            BlockAttention(D, attn_heads, dropout) if use_attention else None
        )

        # ── Output ───────────────────────────────────────────────────
        self.block_weights = nn.Parameter(torch.ones(N) / N)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D * 2),
            _get_activation(activation),
            nn.Linear(D * 2, output_dim),
        )

        # ── Ring indices ─────────────────────────────────────────────
        self.register_buffer("_prev_idx", torch.tensor(
            [(i - 1) % N for i in range(N)], dtype=torch.long
        ))
        self.register_buffer("_opp_idx", torch.tensor(
            [(i + N // 2) % N for i in range(N)], dtype=torch.long
        ))

    def _ring_step(self, states: torch.Tensor) -> torch.Tensor:
        """Single ring update. states: (N, B, D) → (N, B, D)."""
        N = states.shape[0]
        prev = states[self._prev_idx]
        opp = states[self._opp_idx]
        hedge = opp - prev

        if isinstance(self.blocks[0], AdaptiveRingBlock):
            updates = torch.stack([
                self.blocks[i](prev[i], hedge[i]) - prev[i]
                for i in range(N)
            ])
        else:
            updates = torch.stack([
                self.blocks[i](torch.cat([prev[i], hedge[i]], dim=-1))
                for i in range(N)
            ])

        gamma = self.damping
        return (1 - gamma) * states + gamma * (prev + updates)

    def forward(
        self,
        x: torch.Tensor,
        return_all_cycles: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        B = x.shape[0]
        N, D = self.num_blocks, self.ring_dim

        # Input projection
        if self.use_conv_stem and x.dim() == 4:
            base = self.conv_stem(x)  # (B, C, H, W) → (B, D)
        elif x.dim() > 2:
            base = self.shared_proj(x.view(B, -1))
        else:
            base = self.shared_proj(x)

        states = base.unsqueeze(0) + self.block_seeds.unsqueeze(1)

        all_cycles: list[torch.Tensor] | None = [states] if return_all_cycles else None

        for _ in range(self.num_cycles):
            states = self._ring_step(states)
            if self.attention is not None:
                states = self.attention(states)
            if return_all_cycles:
                assert all_cycles is not None
                all_cycles.append(states)

        w = F.softmax(self.block_weights, dim=0)
        aggregated = (states * w.view(N, 1, 1)).sum(dim=0)
        out = self.output_proj(aggregated)

        if return_all_cycles:
            return out, all_cycles
        return out

    def get_block_importance(self) -> torch.Tensor:
        return F.softmax(self.block_weights, dim=0)

    def get_adaptive_alphas(self) -> torch.Tensor | None:
        """Return learned alpha values (only for adaptive_residual mode)."""
        if isinstance(self.blocks[0], AdaptiveRingBlock):
            return torch.stack([b.alpha.data for b in self.blocks])
        return None

    def extra_repr(self) -> str:
        return (
            f"ring_dim={self.ring_dim}, num_blocks={self.num_blocks}, "
            f"num_cycles={self.num_cycles}, damping={self.damping}"
        )
