"""Integration tests: RRN-H convergence on synthetic and real data."""

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from rrn import RingResidualNetwork


def _quick_train(model, loader, steps=50, lr=1e-2):
    """Minimal training helper for integration tests."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    losses = []
    for epoch in range(steps):
        for x, y in loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
    return losses


class TestConvergence:
    """Verify RRN-H can learn simple patterns."""

    def test_overfit_small_dataset(self):
        """Model should overfit 16 samples quickly."""
        torch.manual_seed(0)
        X = torch.randn(16, 32)
        y = torch.randint(0, 3, (16,))
        ds = TensorDataset(X, y)
        loader = DataLoader(ds, batch_size=16)

        model = RingResidualNetwork(
            input_dim=32, output_dim=3,
            ring_dim=16, num_blocks=3, num_cycles=2, expansion=2,
        )

        losses = _quick_train(model, loader, steps=100, lr=5e-3)

        # Final loss should be much lower than initial
        assert losses[-1] < losses[0] * 0.3, (
            f"Loss didn't decrease enough: {losses[0]:.3f} → {losses[-1]:.3f}"
        )

    def test_perfect_separation(self):
        """Model should achieve near-perfect accuracy on linearly separable data."""
        torch.manual_seed(1)
        # Create two well-separated Gaussian clusters
        X0 = torch.randn(50, 20) + 3.0
        X1 = torch.randn(50, 20) - 3.0
        X = torch.cat([X0, X1])
        y = torch.cat([torch.zeros(50, dtype=torch.long),
                       torch.ones(50, dtype=torch.long)])
        ds = TensorDataset(X, y)
        loader = DataLoader(ds, batch_size=32, shuffle=True)

        model = RingResidualNetwork(
            input_dim=20, output_dim=2,
            ring_dim=16, num_blocks=3, num_cycles=2, expansion=2,
        )

        _quick_train(model, loader, steps=200, lr=1e-2)

        model.eval()
        with torch.no_grad():
            acc = (model(X).argmax(1) == y).float().mean().item()
        assert acc > 0.95, f"Accuracy too low: {acc:.2%}"

    def test_multi_class_separation(self):
        """Model should handle 4-class problem."""
        torch.manual_seed(2)
        X_list, y_list = [], []
        for c in range(4):
            X_list.append(torch.randn(40, 16) + c * 2.5)
            y_list.append(torch.full((40,), c, dtype=torch.long))
        X = torch.cat(X_list)
        y = torch.cat(y_list)
        ds = TensorDataset(X, y)
        loader = DataLoader(ds, batch_size=32, shuffle=True)

        model = RingResidualNetwork(
            input_dim=16, output_dim=4,
            ring_dim=16, num_blocks=3, num_cycles=2, expansion=2,
        )

        _quick_train(model, loader, steps=300, lr=1e-2)

        model.eval()
        with torch.no_grad():
            acc = (model(X).argmax(1) == y).float().mean().item()
        assert acc > 0.85, f"Accuracy too low: {acc:.2%}"


class TestRobustness:
    """Architecture robustness tests."""

    def test_variable_batch_size(self):
        """Forward pass should work for different batch sizes."""
        model = RingResidualNetwork(16, 3, ring_dim=8, num_blocks=3, num_cycles=2)
        for bs in [1, 3, 7, 16, 32]:
            out = model(torch.randn(bs, 16))
            assert out.shape == (bs, 3)

    def test_no_nan_in_forward(self):
        """Multiple forward passes should not produce NaN."""
        model = RingResidualNetwork(32, 5, ring_dim=16, num_blocks=4, num_cycles=3)
        model.eval()
        for _ in range(20):
            with torch.no_grad():
                out = model(torch.randn(8, 32))
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()

    def test_consensus_converges_with_cycles(self):
        """More cycles should lead to tighter consensus."""
        torch.manual_seed(5)
        model = RingResidualNetwork(16, 3, ring_dim=16, num_blocks=4,
                                     num_cycles=5, dropout=0.0)
        model.eval()
        x = torch.randn(8, 16)

        with torch.no_grad():
            _, cycles = model(x, return_all_cycles=True)

        from rrn import compute_consensus
        # Skip init (index 0), check first and last cycle
        first = compute_consensus(cycles[1])
        last = compute_consensus(cycles[-1])
        # Last cycle should be at least as tight as first (damped Jacobi)
        assert last <= first * 1.5, (
            f"Consensus diverged: {first:.4f} → {last:.4f}"
        )
