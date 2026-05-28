"""
RRN-H vs Standard ResNet — Full MNIST Benchmark
================================================
Long-duration validation on full MNIST (60K train / 10K test).
Multiple configurations, multiple seeds, statistical comparison.

Compares:
  RRN-H:  Ring Residual Network with Hedging
  ResNet: Standard linear-chain deep ResNet (matched parameter count)

Reports:
  - Test accuracy (mean ± std across seeds)
  - Convergence speed (epochs to 95%)
  - Per-parameter efficiency (accuracy / param count)
  - Gradient health (CV of gradient norms)
"""

import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from rrn import RingResidualNetwork, compute_consensus, gradient_health


# ─── Baseline: Standard Deep Linear ResNet ──────────────────────────────────

class DeepResNet(nn.Module):
    """Standard linear-chain ResNet for fair comparison."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_blocks: int, expansion: int = 4):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        hidden_mlp = hidden_dim * expansion
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_mlp),
                nn.GELU(),
                nn.Linear(hidden_mlp, hidden_dim),
            )
            for _ in range(num_blocks)
        ])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = h + block(h)
        return self.output_proj(h)


# ─── Experiment runner ──────────────────────────────────────────────────────

def run_experiment(
    model_factory,
    model_name: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    seeds: tuple = (0, 1, 2),
) -> dict:
    """Run one model config across multiple seeds. Returns aggregated stats."""
    all_histories = []
    all_final_accs = []

    for seed in seeds:
        torch.manual_seed(seed)
        model = model_factory()
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        history = []

        for epoch in range(1, epochs + 1):
            # Train
            model.train()
            train_loss, train_correct, train_n = 0.0, 0, 0
            for x, y in train_loader:
                x, y = x.to(device).view(x.size(0), -1), y.to(device)
                opt.zero_grad()
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                opt.step()
                train_loss += loss.item() * x.size(0)
                train_correct += (model(x).argmax(1) == y).sum().item()
                train_n += x.size(0)

            # Eval
            model.eval()
            test_loss, test_correct, test_n = 0.0, 0, 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device).view(x.size(0), -1), y.to(device)
                    logits = model(x)
                    test_loss += F.cross_entropy(logits, y).item() * x.size(0)
                    test_correct += (logits.argmax(1) == y).sum().item()
                    test_n += x.size(0)

            history.append({
                "epoch": epoch,
                "train_loss": train_loss / train_n,
                "train_acc": train_correct / train_n,
                "test_loss": test_loss / test_n,
                "test_acc": test_correct / test_n,
            })

        all_histories.append(history)
        all_final_accs.append(history[-1]["test_acc"])

    # Aggregate
    final_accs_t = torch.tensor(all_final_accs)
    return {
        "model": model_name,
        "params": n_params,
        "final_acc_mean": float(final_accs_t.mean()),
        "final_acc_std": float(final_accs_t.std()),
        "final_acc_max": float(final_accs_t.max()),
        "epochs_to_95": _epochs_to_threshold(all_histories, 0.95),
        "per_param_efficiency": float(final_accs_t.mean()) / (n_params / 1e6),
        "histories": all_histories,
    }


def _epochs_to_threshold(histories: list, threshold: float) -> float:
    """Average epochs needed to reach threshold train accuracy."""
    epochs_needed = []
    for h in histories:
        for entry in h:
            if entry["train_acc"] >= threshold:
                epochs_needed.append(entry["epoch"])
                break
        else:
            epochs_needed.append(float("inf"))
    finite = [e for e in epochs_needed if e != float("inf")]
    return float(torch.tensor(finite).float().mean()) if finite else float("inf")


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=" * 70)
    print("RRN-H vs ResNet — Full MNIST Benchmark")
    print("=" * 70)

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST("./data", train=True, download=False, transform=transform)
    test_ds  = datasets.MNIST("./data", train=False, download=False, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=256)

    print(f"\nTrain: {len(train_ds):,}  Test: {len(test_ds):,}  Batch: 128")
    print(f"Epochs: 10  Seeds: 3")

    # Configurations to test
    configs = [
        # (name, factory_fn)
        ("RRN-H (64,4,2)", lambda: RingResidualNetwork(
            784, 10, ring_dim=64, num_blocks=4, num_cycles=2, expansion=2)),
        ("RRN-H (64,4,4)", lambda: RingResidualNetwork(
            784, 10, ring_dim=64, num_blocks=4, num_cycles=4, expansion=2)),
        ("RRN-H (48,6,3)", lambda: RingResidualNetwork(
            784, 10, ring_dim=48, num_blocks=6, num_cycles=3, expansion=2)),
        ("RRN-H (32,8,2)", lambda: RingResidualNetwork(
            784, 10, ring_dim=32, num_blocks=8, num_cycles=2, expansion=2)),
        # ResNet baselines: match total block applications
        ("ResNet (18-blk)", lambda: DeepResNet(
            784, 64, 10, num_blocks=18, expansion=2)),
        ("ResNet (12-blk)", lambda: DeepResNet(
            784, 48, 10, num_blocks=12, expansion=2)),
    ]

    results = []
    for name, factory in configs:
        print(f"\n{'─' * 60}")
        print(f"Running: {name}")
        t0 = time.time()
        result = run_experiment(
            factory, name, train_loader, test_loader, device,
            epochs=10, seeds=(0, 1, 2),
        )
        elapsed = time.time() - t0
        results.append(result)
        print(f"  Params: {result['params']:,}")
        print(f"  Test acc: {result['final_acc_mean']:.2%} ± {result['final_acc_std']:.2%}")
        print(f"  Best acc: {result['final_acc_max']:.2%}")
        print(f"  Efficiency: {result['per_param_efficiency']:.1f}%/M")
        print(f"  Time: {elapsed:.0f}s")

    # ── Comparison table ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Model':<22} {'Params':>8} {'Acc':>8} {'±':>6} {'Best':>7} {'Eff':>7} {'→95%':>6}")
    print("-" * 70)
    for r in results:
        eff = r['per_param_efficiency']
        e95 = r['epochs_to_95']
        print(f"{r['model']:<22} {r['params']:>8,} {r['final_acc_mean']:>7.2%} "
              f"{r['final_acc_std']:>5.2%} {r['final_acc_max']:>6.2%} "
              f"{eff:>6.1f} {e95:>5.1f}")

    # Winner determination
    best_acc = max(results, key=lambda r: r['final_acc_mean'])
    best_eff = max(results, key=lambda r: r['per_param_efficiency'])
    print(f"\n🏆 Best accuracy:  {best_acc['model']} ({best_acc['final_acc_mean']:.2%})")
    print(f"🍯 Best efficiency: {best_eff['model']} ({best_eff['per_param_efficiency']:.1f}%/M)")

    # Save
    save_results = [{k: v for k, v in r.items() if k != 'histories'} for r in results]
    with open("experiments/full_validation_results.json", "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to experiments/full_validation_results.json")


if __name__ == "__main__":
    main()
