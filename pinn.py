"""
Physics-Informed Neural Network (PINN) for the Damped Harmonic Oscillator

Solves:  x''(t) + 2*gamma*x'(t) + omega_0^2 * x(t) = 0
ICs:     x(0) = 1,  x'(0) = 0

The network learns the solution x(t) by minimizing the residual of the ODE
at collocation points, without any labeled data -- just the physics.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Physics parameters
# ---------------------------------------------------------------------------
GAMMA = 0.25        # damping coefficient
OMEGA_0 = 2.0       # natural frequency
T_MAX = 15.0        # simulation time
IC_X0 = 1.0         # initial displacement
IC_V0 = 0.0         # initial velocity


# ---------------------------------------------------------------------------
# Analytical solution (for validation)
# ---------------------------------------------------------------------------
def analytical_solution(t):
    """Exact solution of the underdamped harmonic oscillator."""
    omega_d = np.sqrt(OMEGA_0**2 - GAMMA**2)
    A = IC_X0
    B = (IC_V0 + GAMMA * IC_X0) / omega_d
    return np.exp(-GAMMA * t) * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------
class PINN(nn.Module):
    """Simple fully-connected network: t -> x(t)."""

    def __init__(self, hidden=64, layers=4):
        super().__init__()
        modules = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            modules += [nn.Linear(hidden, hidden), nn.Tanh()]
        modules.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*modules)

    def forward(self, t):
        return self.net(t)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
def physics_loss(model, t_colloc):
    """ODE residual: x'' + 2*gamma*x' + omega_0^2 * x = 0."""
    t = t_colloc.clone().requires_grad_(True)
    x = model(t)

    dx_dt = torch.autograd.grad(
        x, t, grad_outputs=torch.ones_like(x), create_graph=True
    )[0]

    d2x_dt2 = torch.autograd.grad(
        dx_dt, t, grad_outputs=torch.ones_like(dx_dt), create_graph=True
    )[0]

    residual = d2x_dt2 + 2 * GAMMA * dx_dt + OMEGA_0**2 * x
    return torch.mean(residual**2)


def ic_loss(model):
    """Enforce x(0) = x0 and x'(0) = v0."""
    t0 = torch.zeros(1)
    x_pred = model(t0)

    x_loss = (x_pred.squeeze() - IC_X0) ** 2

    t0_grad = t0.clone().requires_grad_(True)
    x_pred_grad = model(t0_grad)
    dx_dt = torch.autograd.grad(
        x_pred_grad, t0_grad, grad_outputs=torch.ones_like(x_pred_grad),
        create_graph=True
    )[0]
    v_loss = (dx_dt.squeeze() - IC_V0) ** 2

    return x_loss + v_loss


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(model, device, n_colloc=400, n_iters=15000, lr=1e-3, print_every=500):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iters)

    t_colloc = torch.linspace(0, T_MAX, n_colloc).unsqueeze(1).to(device)

    history = {"total": [], "physics": [], "ic": []}

    for i in range(1, n_iters + 1):
        optimizer.zero_grad()

        l_physics = physics_loss(model, t_colloc)
        l_ic = ic_loss(model)
        loss = l_physics + 10.0 * l_ic

        loss.backward()
        optimizer.step()
        scheduler.step()

        history["total"].append(loss.item())
        history["physics"].append(l_physics.item())
        history["ic"].append(l_ic.item())

        if i % print_every == 0 or i == 1:
            print(
                f"  step {i:>5d}/{n_iters}  |  "
                f"total: {loss.item():.6f}  physics: {l_physics.item():.6f}  "
                f"ic: {l_ic.item():.6f}"
            )

    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, device, n_pts=500):
    model.eval()
    t_np = np.linspace(0, T_MAX, n_pts)
    t_torch = torch.tensor(t_np, dtype=torch.float32, device=device).unsqueeze(1)

    with torch.no_grad():
        x_pred = model(t_torch).cpu().numpy().squeeze()

    x_exact = analytical_solution(t_np)
    error = np.abs(x_pred - x_exact)

    return t_np, x_pred, x_exact, error


# ---------------------------------------------------------------------------
# Benchmarking (stretch goal: GPU vs CPU)
# ---------------------------------------------------------------------------
def benchmark(device_str, n_iters=2000):
    device = torch.device(device_str)
    print(f"\n--- Benchmark on {device} ({n_iters} iters) ---")

    model = PINN(hidden=64, layers=4)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    t_colloc = torch.linspace(0, T_MAX, 200).unsqueeze(1).to(device)

    # warmup
    for _ in range(5):
        optimizer.zero_grad()
        l = physics_loss(model, t_colloc) + 10.0 * ic_loss(model)
        l.backward()
        optimizer.step()

    torch.cuda.synchronize() if device.type == "cuda" else None
    t_start = time.perf_counter()

    for _ in range(n_iters):
        optimizer.zero_grad()
        l = physics_loss(model, t_colloc) + 10.0 * ic_loss(model)
        l.backward()
        optimizer.step()

    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.perf_counter() - t_start

    print(f"  {n_iters} training steps completed in {elapsed:.3f}s "
          f"({elapsed / n_iters * 1000:.2f} ms/step)")
    return elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PINN for damped harmonic oscillator")
    parser.add_argument("--iters", type=int, default=15000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--colloc", type=int, default=400)
    parser.add_argument("--benchmark", action="store_true",
                        help="Run GPU vs CPU timing comparison")
    parser.add_argument("--save", action="store_true",
                        help="Save trained model and results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Train
    print(f"\nTraining PINN ({args.iters} iterations)...")
    model = PINN(hidden=args.hidden, layers=args.layers)
    history = train(model, device, n_colloc=args.colloc,
                    n_iters=args.iters, lr=args.lr)

    # Evaluate
    print("\nEvaluating...")
    t_np, x_pred, x_exact, error = evaluate(model, device)
    max_err = error.max()
    mean_err = error.mean()
    print(f"  Max absolute error:  {max_err:.6f}")
    print(f"  Mean absolute error: {mean_err:.6f}")

    # Save results
    if args.save:
        out = Path("results")
        out.mkdir(exist_ok=True)

        torch.save(model.state_dict(), out / "model.pt")
        np.savez(out / "predictions.npz",
                 t=t_np, x_pred=x_pred, x_exact=x_exact, error=error,
                 history_total=history["total"],
                 history_physics=history["physics"],
                 history_ic=history["ic"])
        print(f"\nResults saved to {out}/")

        # Generate and save plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            axes[0].plot(t_np, x_exact, "b-", label="Analytical", linewidth=2)
            axes[0].plot(t_np, x_pred, "r--", label="PINN", linewidth=2)
            axes[0].set_xlabel("t")
            axes[0].set_ylabel("x(t)")
            axes[0].set_title("Solution Comparison")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            axes[1].semilogy(t_np, error + 1e-10)
            axes[1].set_xlabel("t")
            axes[1].set_ylabel("|Error|")
            axes[1].set_title("Absolute Error")
            axes[1].grid(True, alpha=0.3)

            axes[2].semilogy(history["total"], label="Total")
            axes[2].semilogy(history["physics"], label="Physics")
            axes[2].semilogy(history["ic"], label="IC")
            axes[2].set_xlabel("Training step")
            axes[2].set_ylabel("Loss")
            axes[2].set_title("Training Loss")
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(out / "results.png", dpi=150, bbox_inches="tight")
            print(f"Plot saved to {out}/results.png")
        except ImportError:
            print("matplotlib not found -- skipping plot generation")

    # Benchmark stretch goal
    if args.benchmark:
        times = {}
        times["cpu"] = benchmark("cpu", n_iters=2000)
        if torch.cuda.is_available():
            times["gpu"] = benchmark("cuda", n_iters=2000)
            speedup = times["cpu"] / times["gpu"]
            print(f"\nGPU speedup: {speedup:.2f}x")
        else:
            print("\nNo CUDA GPU detected -- skipping GPU benchmark.")


if __name__ == "__main__":
    main()
