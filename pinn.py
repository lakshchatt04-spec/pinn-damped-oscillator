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

try:
    from scipy.integrate import solve_ivp
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

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
    """Simple fully-connected network: t -> x(t).

    If hard_ic is True, the output is reparameterized as
        x(t) = x0 + v0*t + t^2 * N(t)
    which satisfies the initial conditions exactly for any network N(t).
    Otherwise the ICs are imposed as a soft penalty (see ic_loss).
    """

    def __init__(self, hidden=64, layers=4, hard_ic=False):
        super().__init__()
        self.hard_ic = hard_ic
        modules = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            modules += [nn.Linear(hidden, hidden), nn.Tanh()]
        modules.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*modules)

    def forward(self, t):
        n = self.net(t)
        if self.hard_ic:
            return IC_X0 + IC_V0 * t + t**2 * n
        return n


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


def set_seed(seed):
    """Make a run reproducible."""
    torch.manual_seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(model, device, n_colloc=400, n_iters=15000, lr=1e-3, print_every=500,
          ic_weight=10.0, seed=None):
    if seed is not None:
        set_seed(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iters)

    t_colloc = torch.linspace(0, T_MAX, n_colloc).unsqueeze(1).to(device)

    history = {"total": [], "physics": [], "ic": []}

    for i in range(1, n_iters + 1):
        optimizer.zero_grad()

        l_physics = physics_loss(model, t_colloc)
        l_ic = ic_loss(model)
        loss = l_physics + ic_weight * l_ic

        loss.backward()
        optimizer.step()
        scheduler.step()

        history["total"].append(loss.item())
        history["physics"].append(l_physics.item())
        history["ic"].append(l_ic.item())

        if print_every and (i % print_every == 0 or i == 1):
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

    # Initial-condition errors computed exactly via autograd at t=0
    # (not from the sampling grid, to avoid interpolation artifacts).
    t0 = torch.zeros(1, 1, device=device, requires_grad=True)
    x0 = model(t0)
    v0 = torch.autograd.grad(
        x0, t0, grad_outputs=torch.ones_like(x0), create_graph=False
    )[0]
    ic_error_x = float(abs(x0.item() - IC_X0))
    ic_error_v = float(abs(v0.item() - IC_V0))

    return t_np, x_pred, x_exact, error, ic_error_x, ic_error_v


# ---------------------------------------------------------------------------
# Numerical baselines: RK4 and SciPy solve_ivp
# ---------------------------------------------------------------------------
def ode_system(t, x_vec):
    """State: [x, v]. Return [v, -2*GAMMA*v - OMEGA_0^2*x]."""
    x, v = x_vec
    dx = v
    dv = -2.0 * GAMMA * v - OMEGA_0**2 * x
    return np.array([dx, dv])


def rk4_solution(t_eval):
    """Compute numerical solution using 4th-order Runge-Kutta."""
    t_eval = np.asarray(t_eval, dtype=float)
    if len(t_eval) == 0:
        return t_eval, np.zeros(0)
    t0 = t_eval[0]
    tf = t_eval[-1]
    h = t_eval[1] - t_eval[0] if len(t_eval) > 1 else (tf - t0) / 1000.0
    # If t_eval is dense and spans whole domain, use uniform h; otherwise fallback to simple stepping
    x = np.zeros_like(t_eval, dtype=float)
    v = np.zeros_like(t_eval, dtype=float)
    x_cur, v_cur = IC_X0, IC_V0
    x[0], v[0] = x_cur, v_cur
    for i in range(1, len(t_eval)):
        ti = t_eval[i - 1]
        ti1 = t_eval[i]
        h_i = ti1 - ti
        k1x, k1v = ode_system(ti, np.array([x_cur, v_cur]))
        k2x, k2v = ode_system(ti + h_i / 2.0, np.array([x_cur + h_i / 2.0 * k1x, v_cur + h_i / 2.0 * k1v]))
        k3x, k3v = ode_system(ti + h_i / 2.0, np.array([x_cur + h_i / 2.0 * k2x, v_cur + h_i / 2.0 * k2v]))
        k4x, k4v = ode_system(ti + h_i, np.array([x_cur + h_i * k3x, v_cur + h_i * k3v]))
        x_cur = x_cur + h_i / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        v_cur = v_cur + h_i / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        x[i], v[i] = x_cur, v_cur
    return t_eval, x


def scipy_solution(t_eval):
    """Compute numerical solution using SciPy solve_ivp."""
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is not available")
    sol = solve_ivp(ode_system, (t_eval[0], t_eval[-1]), (IC_X0, IC_V0), t_eval=t_eval, method='RK45', rtol=1e-9, atol=1e-12)
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    return sol.t, sol.y[0]


def get_numerical_baselines(t_eval):
    results = {}
    try:
        t_rk, x_rk = rk4_solution(t_eval)
        results['rk4'] = {'t': t_rk, 'x': x_rk}
    except Exception as e:
        results['rk4_error'] = str(e)
    try:
        t_sp, x_sp = scipy_solution(t_eval)
        results['scipy'] = {'t': t_sp, 'x': x_sp}
    except Exception as e:
        results['scipy_error'] = str(e)
    return results
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
    parser.add_argument("--ic-weight", type=float, default=10.0,
                        help="Weight on the initial-condition loss term")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility (default 0)")
    parser.add_argument("--hard-ic", action="store_true",
                        help="Hard-enforce ICs via x(t)=x0+v0*t+t^2*N(t)")
    parser.add_argument("--n-eval", type=int, default=2000, help="Evaluation points (denser grid for max error)")
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
    model = PINN(hidden=args.hidden, layers=args.layers, hard_ic=args.hard_ic)
    history = train(model, device, n_colloc=args.colloc,
                    n_iters=args.iters, lr=args.lr,
                    ic_weight=args.ic_weight, seed=args.seed)

    # Evaluate
    print("\nEvaluating...")
    t_np, x_pred, x_exact, error, ic_error_x, ic_error_v = evaluate(model, device, n_pts=args.n_eval)
    max_err = error.max()
    mean_err = error.mean()
    print(f"  Max absolute error:  {max_err:.6f}")
    print(f"  Mean absolute error: {mean_err:.6f}")
    print(f"  IC error x(0):      {ic_error_x:.6f}")
    print(f"  IC error v(0) (fd):  {ic_error_v:.6f}")

    # Save results
    if args.save:
        out = Path("results")
        out.mkdir(exist_ok=True)

        torch.save(model.state_dict(), out / "model.pt")
        np.savez(out / "predictions.npz",
                 t=t_np, x_pred=x_pred, x_exact=x_exact, error=error,
                 ic_error_x=ic_error_x, ic_error_v=ic_error_v,
                 history_total=history["total"],
                 history_physics=history["physics"],
                 history_ic=history["ic"])
        print(f"\nResults saved to {out}/")

        # Generate and save plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 4, figsize=(18, 4))

            axes[0].plot(t_np, x_exact, "b-", label="Analytical", linewidth=2)
            axes[0].plot(t_np, x_pred, "r--", label="PINN", linewidth=2)
            axes[0].set_xlabel("t")
            axes[0].set_ylabel("x(t)")
            axes[0].set_title("Solution Comparison")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # also compare with numerical baselines if available
            num = get_numerical_baselines(t_np)
            if 'scipy' in num:
                axes[0].plot(num['scipy']['t'], num['scipy']['x'], "k:", label="solve_ivp(RK45)", linewidth=1.5)
            elif 'rk4' in num:
                axes[0].plot(num['rk4']['t'], num['rk4']['x'], "k:", label="RK4", linewidth=1.5)
            axes[0].legend()

            axes[1].semilogy(t_np, error + 1e-10)
            axes[1].set_xlabel("t")
            axes[1].set_ylabel("|Error|")
            axes[1].set_title("Absolute Error (PINN vs Analytical)")
            axes[1].grid(True, alpha=0.3)

            # comparison against numerical baseline errors if available
            if 'scipy' in num:
                err_num = np.abs(x_pred - num['scipy']['x'])
                axes[2].semilogy(t_np, err_num + 1e-10, label='PINN vs solve_ivp')
            elif 'rk4' in num:
                err_num = np.abs(x_pred - num['rk4']['x'])
                axes[2].semilogy(t_np, err_num + 1e-10, label='PINN vs RK4')
            else:
                axes[2].text(0.5, 0.5, 'Num. baseline\nunavailable', ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_xlabel("t")
            axes[2].set_ylabel("|Error|")
            axes[2].set_title("PINN vs Numerical")
            axes[2].grid(True, alpha=0.3)
            if 'scipy' in num or 'rk4' in num:
                axes[2].legend()

            axes[3].semilogy(history["total"], label="Total")
            axes[3].semilogy(history["physics"], label="Physics")
            axes[3].semilogy(history["ic"], label="IC")
            axes[3].set_xlabel("Training step")
            axes[3].set_ylabel("Loss")
            axes[3].set_title("Training Loss")
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)

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
