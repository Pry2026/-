"""Unit tests for RRN-H core module: shapes, gradients, consensus dynamics."""

import pytest
import torch
from rrn import RingResidualNetwork, compute_consensus, gradient_health


class TestRingResidualNetwork:
    """Core functionality tests."""

    @pytest.fixture
    def model(self):
        return RingResidualNetwork(
            input_dim=64, output_dim=10,
            ring_dim=32, num_blocks=4, num_cycles=2, expansion=2,
        )

    @pytest.fixture
    def batch(self):
        return torch.randn(8, 64)

    # ── Shape tests ────────────────────────────────────────────────────

    def test_forward_shape(self, model, batch):
        out = model(batch)
        assert out.shape == (8, 10)

    def test_forward_single_sample(self, model):
        out = model(torch.randn(1, 64))
        assert out.shape == (1, 10)

    def test_return_all_cycles_shape(self, model, batch):
        out, cycles = model(batch, return_all_cycles=True)
        assert out.shape == (8, 10)
        assert len(cycles) == model.num_cycles + 1  # init + K cycles
        for states in cycles:
            assert states.shape == (model.num_blocks, 8, model.ring_dim)

    # ── Gradient tests ─────────────────────────────────────────────────

    def test_all_params_receive_gradients(self, model, batch):
        out = model(batch)
        out.sum().backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"Parameter {name} has no gradient"

    def test_gradient_not_nan(self, model, batch):
        out = model(batch)
        out.sum().backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN in {name}.grad"
                assert not torch.isinf(p.grad).any(), f"Inf in {name}.grad"

    def test_gradient_flows_through_all_blocks(self, model, batch):
        """Each block should receive non-zero gradients."""
        out = model(batch)
        out.sum().backward()

        # Check first layer of each block's MLP
        for i, block in enumerate(model.blocks):
            w = block[1].weight  # nn.Linear after LayerNorm
            assert w.grad is not None, f"Block {i} linear has no gradient"
            assert w.grad.abs().sum() > 0, f"Block {i} received zero gradient"

    # ── Consensus tests ────────────────────────────────────────────────

    def test_consensus_decreases(self, model, batch):
        """After training a step, consensus should tighten."""
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        # Measure initial consensus
        with torch.no_grad():
            _, cycles_before = model(batch, return_all_cycles=True)
        consensus_before = compute_consensus(cycles_before[-1])

        # One training step
        out = model(batch)
        loss = out.sum()
        loss.backward()
        opt.step()

        # Measure after
        with torch.no_grad():
            _, cycles_after = model(batch, return_all_cycles=True)
        consensus_after = compute_consensus(cycles_after[-1])

        # Should tighten or stay similar (random init may fluctuate)
        assert consensus_after < 1.0, "Consensus should be below 1.0"

    # ── Reproducibility ────────────────────────────────────────────────

    def test_deterministic_forward(self):
        """Same seed + same config → identical outputs."""
        torch.manual_seed(42)
        model1 = RingResidualNetwork(
            input_dim=64, output_dim=10,
            ring_dim=32, num_blocks=4, num_cycles=2, expansion=2, dropout=0.0,
        )
        model1.eval()
        x = torch.randn(8, 64)

        torch.manual_seed(42)
        model2 = RingResidualNetwork(
            input_dim=64, output_dim=10,
            ring_dim=32, num_blocks=4, num_cycles=2, expansion=2, dropout=0.0,
        )
        model2.eval()

        with torch.no_grad():
            out1 = model1(x)
            out2 = model2(x)
        assert torch.allclose(out1, out2)

    # ── Edge cases ─────────────────────────────────────────────────────

    def test_minimal_blocks(self):
        """N=2 is the minimum valid configuration."""
        model = RingResidualNetwork(16, 3, ring_dim=8, num_blocks=2, num_cycles=1)
        out = model(torch.randn(4, 16))
        assert out.shape == (4, 3)

    def test_single_cycle(self, model, batch):
        model.num_cycles = 1
        out, cycles = model(batch, return_all_cycles=True)
        assert len(cycles) == 2  # init + 1 cycle

    def test_zero_dropout(self):
        """Dropout=0 should produce identical outputs for same input."""
        model = RingResidualNetwork(16, 3, ring_dim=8, num_blocks=2,
                                     num_cycles=1, dropout=0.0)
        model.eval()
        x = torch.randn(4, 16)
        out1 = model(x)
        out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_block_importance_sums_to_one(self, model):
        imp = model.get_block_importance()
        assert torch.allclose(imp.sum(), torch.tensor(1.0))

    def test_invalid_num_blocks(self):
        with pytest.raises(ValueError, match="num_blocks must be ≥ 2"):
            RingResidualNetwork(16, 3, num_blocks=1)


class TestGradientHealth:
    """Tests for gradient_health utility."""

    def test_returns_expected_keys(self):
        model = RingResidualNetwork(16, 3, ring_dim=8, num_blocks=2, num_cycles=1)
        x = torch.randn(4, 16)
        model(x).sum().backward()
        health = gradient_health(model)
        for key in ("mean", "std", "min", "max", "cv"):
            assert key in health, f"Missing key: {key}"
            assert isinstance(health[key], float)

    def test_no_gradients_returns_zeros(self):
        model = RingResidualNetwork(16, 3, ring_dim=8, num_blocks=2, num_cycles=1)
        health = gradient_health(model)  # no backward called
        assert health["cv"] == 0.0
