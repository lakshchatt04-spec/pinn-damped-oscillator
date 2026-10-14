# Physics-Informed Neural Network (PINN) for the Damped Harmonic Oscillator

A physics-informed neural network that solves the damped harmonic oscillator ODE **without any labeled data** — the network learns the solution purely from the equations of motion and initial conditions.

## The Physics

A damped harmonic oscillator (mass on a spring with friction) obeys:

```
x''(t) + 2·γ·x'(t) + ω₀²·x(t) = 0
x(0)  = 1        (initial displacement)
x'(0) = 0        (initial velocity)
```

With parameters γ = 0.25 (damping) and ω₀ = 2.0 (natural frequency), the system is **underdamped** and oscillates with an amplitude that decays exponentially. The exact analytical solution is:

```
x(t) = e^(−γt) · [ A·cos(ω_d·t) + B·sin(ω_d·t) ]
      where ω_d = √(ω₀² − γ²)
```

## The Approach

Instead of numerically stepping the ODE forward in time (like Euler or Runge-Kutta), a PINN bakes the physics directly into the loss function:

| Loss term | What it enforces |
|-----------|------------------|
| **Physics residual** | The network output `x(t)` must satisfy the ODE at collocation points across the time domain. Autograd computes `x''` and `x'` from the network. |
| **Initial condition** | `x(0) = 1` and `x'(0) = 0` are enforced as *soft* constraints (weighted more heavily than the residual). |

Total loss = `physics_residual² + 10·(IC terms)`. Training minimizes this loss with Adam + cosine annealing.

*(Note: I also experimented with hard-enforcing the ICs via the ansatz `x(t) = x0 + v0·t + t²·N(t)`. It guarantees the ICs exactly, but the `t²` scaling makes the network predict tiny amplitudes over a wide dynamic range and actually converged worse here — a nice lesson in how solution parameterization affects PINN training.)*

The key idea: the network is a surrogate for the solution function `x(t)`, and automatic differentiation (autograd) lets us impose the differential equation directly — no discretization grid required.

## Results

Comparison of the trained PINN against the exact analytical solution:

- **Max absolute error**: ~`6e-2` (over t ∈ [0, 15])
- **Mean absolute error**: ~`2e-2`

The network captures both the exponential decay envelope and the damped oscillations across the full time domain. (With more training time and a wider/deeper net the error drops further; the numbers above are from the default `pinn.py --save` run.)

![Results](results/results.png)

*(Plots: solution comparison, absolute error, training loss.) Generate them yourself with `python pinn.py --save`.*

## Stretch Goal: GPU vs CPU Benchmarking

Run `python pinn.py --benchmark` to time identical training loops on CPU versus GPU (if a CUDA GPU is available). This puts concrete numbers behind the "how much does hardware matter?" question — great talking point for interviews.

## Getting Started

```bash
# 1. Clone / enter the project
git clone <your-repo-url> && cd pinn-damped-oscillator

# 2. (Recommended) create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model, evaluate vs analytical, and save plots
python pinn.py --save

# 5. (Optional) benchmark CPU vs GPU
python pinn.py --benchmark
```

Tested with PyTorch 2.x and Python 3.10+.

## Project Layout

```
pinn-damped-oscillator/
├── pinn.py            # Full implementation: model, loss, training, eval, benchmark
├── requirements.txt   # torch, numpy, matplotlib
├── README.md
└── results/           # (generated) model.pt, predictions.npz, results.png
```

## Why this project is useful

This project combines neural networks with a physical system whose behavior can be calculated analytically. By training a physics-informed neural network (PINN) to solve the damped harmonic oscillator, I can investigate how the choice of loss function, training parameters, and differential-equation constraints affects the accuracy of the predicted solution.

Comparing the network's predictions with the analytical solution provides a direct way to measure its accuracy and understand where the approach succeeds or struggles.

## Next Steps / Ideas

- Try a harder equation: the **nonlinear** pendulum `θ'' + (g/L)·sin(θ) = 0` (no closed-form solution — a true test).
- Explore better IC handling (e.g. a Fourier-feature or input-scaled ansatz that avoids the large-dynamic-range problem).
- Compare against a standard numerical solver (RK4) for accuracy across different damping regimes.
