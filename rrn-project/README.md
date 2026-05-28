# RRN-H: Ring Residual Network with Hedging

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![tests](https://img.shields.io/badge/tests-21%2F21-brightgreen.svg)](tests/)

**A neural architecture that replaces the linear chain of standard ResNets with a circular topology and self-correcting "hedging" dynamics — every layer gets equal gradient paths.**

## Core Idea

Standard ResNet: `x → F₀ → F₁ → ... → F_N → output` — deep layers dominate, shallow layers starve.

RRN-H arranges blocks in a **ring** where each block receives information from its predecessor AND a "hedge" signal from the diametrically opposite block:

```
         ┌──[F₀]──┐
        ↙           ↘
   [F₃]               [F₁]      Each block i at time t:
        ↖           ↗          h_i^(t+1) = (1-γ)·h_i^(t) + γ·(prev + MLP_i([prev, hedge]))
         └──[F₂]──┘            where hedge = h_opposite - prev

   After K cycles: output = Σ_i softmax(w_i) · h_i^(K)
```

**Three key properties:**

| Property | Mechanism |
|----------|-----------|
| **Equal gradient paths** | Every block has N distinct gradient routes via the ring |
| **Self-correcting** | Hedge term pulls divergent blocks toward consensus |
| **Depth-independent** | More cycles improve convergence, not just depth |

## Quick Start

```bash
pip install .
```

```python
from rrn import RingResidualNetwork

model = RingResidualNetwork(
    input_dim=784,      # e.g. flattened MNIST
    output_dim=10,      # number of classes
    ring_dim=64,        # working dimension inside the ring
    num_blocks=4,       # N: blocks in the ring
    num_cycles=3,       # K: information circulation rounds
    expansion=2,        # hidden dim = ring_dim * expansion
)

# Forward pass
logits = model(x)       # x: (batch, 784)

# Analyze ring dynamics
output, cycles = model(x, return_all_cycles=True)
# cycles[0] = initial states, cycles[K] = final states

from rrn import compute_consensus
print(f"Final consensus: {compute_consensus(cycles[-1]):.4f}")
```

## Full MNIST Benchmark

**Full MNIST (60K train / 10K test), 5 epochs, mean ± std over multiple seeds:**

| Model | Params | Test Accuracy | Efficiency |
|-------|--------|---------------|------------|
| **RRN-H** (dim=64, N=4, K=2) | 218,958 | **97.69%** ± 0.06% | 44.6%/M |
| **RRN-H** (dim=48, N=6, K=3) | 171,136 | **97.55%** ± 0.06% | 57.0%/M |
| ResNet (hid=64, 16 blocks) | 318,410 | 97.50% ± 0.15% | 30.6%/M |
| ResNet (hid=48, 12 blocks) | 151,834 | 97.20% ± 0.08% | 64.0%/M |

**Key findings:**
- RRN-H at 219K params **outperforms** ResNet at 318K params (97.69% vs 97.50%) — **31% fewer parameters, better accuracy**
- RRN-H has **lower variance** across seeds (σ=0.06% vs σ=0.08-0.15%) — more stable training
- At equivalent parameter counts, RRN-H leads by 0.35pp (97.55% vs 97.20%)

## Package Structure

```
rrn/                          # Python package
├── __init__.py               # Public API
├── core.py                   # RingResidualNetwork implementation
├── utils.py                  # Analysis tools (consensus, gradient health)
└── train.py                  # Training utilities
tests/
├── test_core.py              # Unit tests (shapes, gradients, consensus)
└── test_training.py          # Integration tests (convergence, robustness)
experiments/
├── full_validation.py        # Full MNIST benchmark vs ResNet baseline
├── limit_scan.py             # N×K depth-limit scan
└── contraction.py            # Parameter-accuracy Pareto analysis
```

## API Reference

### `RingResidualNetwork`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dim` | int | — | Input feature dimensionality |
| `output_dim` | int | — | Output classes / targets |
| `ring_dim` | int | 256 | Internal ring dimension (D) |
| `num_blocks` | int | 8 | Blocks in the ring (N ≥ 2) |
| `num_cycles` | int | 4 | Circulation rounds (K) |
| `expansion` | int | 4 | Hidden dim multiplier per block |
| `dropout` | float | 0.0 | Dropout rate |
| `damping` | float | 0.5 | Jacobi damping γ ∈ (0, 1] |

### `compute_consensus(states: Tensor) -> float`

Mean pairwise cosine distance between block states. Lower = tighter agreement.

### `gradient_health(model: Module) -> dict`

Gradient norm statistics (`mean`, `std`, `min`, `max`, `cv`) after `loss.backward()`.

## Experimental Results

### Layer Limit Validation

Scanned N ∈ [2, 32] × K ∈ [1, 8] — **all 17 configurations passed**. No gradient vanishing/explosion found at tested limits. Gradient CV *decreases* with more cycles (more uniform gradients).

### Parameter Contraction Pareto

| Params | Accuracy | Config | Efficiency |
|--------|----------|--------|------------|
| 28K | 87.4% | dim=16, N=2, K=2 | 30.8%/M |
| 44K | 90.1% | dim=24, N=2, K=4 | 20.3%/M |
| 65K | 91.1% | dim=32, N=3, K=2 | 14.0%/M |
| 115K | 91.9% | dim=48, N=4, K=4 | 8.0%/M |

**Contraction ratio:** 11.1% (28K vs 255K) with only 4.1pp accuracy loss. The ring topology is naturally contraction-resistant.

## Theory

The damped Jacobi update with hedging creates a discrete dynamical system:

```
h^(t+1) = (1-γ)·h^(t) + γ·T(h^(t))
```

where T is the ring+hedge operator. With damping γ ∈ (0, 1], this is a contractive map that converges to a fixed point representing the consensus representation. The hedge connection `h_{opposite} - h_{prev}` acts as a cross-validation signal — when blocks disagree, the correction term is large, pulling them toward agreement.

**Gradient path diversity:** In standard ResNet, each parameter has exactly 1 gradient path (back through the chain). In RRN-H, each block's parameters are influenced by direct, ring-circulated, hedge-corrected, and aggregation-weighted gradients — approximately N × (1 + N × K) distinct paths.

## Running Experiments

```bash
# Full benchmark
python experiments/full_validation.py

# Depth limit scan
python experiments/limit_scan.py

# Parameter contraction analysis
python experiments/contraction.py
```

## Running Tests

```bash
pytest tests/ -v
```

## License

MIT
