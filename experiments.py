"""
Systematic experiment sweep for the PINN damped-oscillator project.

Varies one design factor at a time around a base configuration, repeating each
setting over multiple random seeds so that a single lucky run cannot drive the
conclusion. Reports max/mean absolute error against the analytical solution and
the initial-condition errors.

Outputs
-------
results/experiments.csv : one row per (factor, value, seed) run
results/experiments.png : error vs each swept factor (mean +/- std over seeds)
"""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np

import pinn


BASE = dict(hidden=64, layers=4, colloc=200, iters=3000, ic_weight=10.0, lr=1e-3,
            hard_ic=False)

SWEEPS = {
    "colloc":    [50, 100, 200, 400],
    "ic_weight": [1.0, 10.0, 100.0],
    "hidden":    [32, 64, 128],
    "iters":     [1000, 3000, 6000],
    "hard_ic":   [False, True],
}


def run_one(device, cfg, seed, n_eval=1000):
    model = pinn.PINN(hidden=cfg["hidden"], layers=cfg["layers"],
                      hard_ic=cfg["hard_ic"])
    pinn.train(model, device, n_colloc=cfg["colloc"], n_iters=cfg["iters"],
               lr=cfg["lr"], print_every=0, ic_weight=cfg["ic_weight"],
               seed=seed)
    _, _, _, error, ic_x, ic_v = pinn.evaluate(model, device, n_pts=n_eval)
    return {
        "max_err": float(error.max()),
        "mean_err": float(error.mean()),
        "ic_err_x": float(ic_x),
        "ic_err_v": float(ic_v),
    }


def main():
    parser = argparse.ArgumentParser(description="PINN experiment sweep")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of random seeds per configuration")
    parser.add_argument("--quick", action="store_true",
                        help="Fewer iterations/configs for a fast smoke test")
    args = parser.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    print(f"Device: {device} | threads: {torch.get_num_threads()} | "
          f"seeds/config: {args.seeds}")

    if args.quick:
        for k in SWEEPS:
            SWEEPS[k] = SWEEPS[k][:2]
        SWEEPS["iters"] = [500, 1000]

    out = Path("results")
    out.mkdir(exist_ok=True)

    rows = []
    total = sum(len(v) for v in SWEEPS.values()) * args.seeds
    done = 0

    for factor, values in SWEEPS.items():
        for value in values:
            cfg = dict(BASE)
            cfg[factor] = value
            for seed in range(args.seeds):
                res = run_one(device, cfg, seed)
                row = {"factor": factor, "value": value, "seed": seed}
                row.update({k: cfg[k] for k in ("hidden", "layers", "colloc",
                                                "iters", "ic_weight", "hard_ic")})
                row.update(res)
                rows.append(row)
                done += 1
                print(f"[{done:>3d}/{total}] {factor}={value!s:<6} seed={seed} "
                      f"-> max={res['max_err']:.4f} mean={res['mean_err']:.4f} "
                      f"ic_x={res['ic_err_x']:.2e} ic_v={res['ic_err_v']:.2e}")

    csv_path = out / "experiments.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} runs to {csv_path}")

    plot_summary(rows, out / "experiments.png")


def plot_summary(rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not found -- skipping experiment plot")
        return

    factors = list(SWEEPS.keys())
    fig, axes = plt.subplots(1, len(factors), figsize=(4 * len(factors), 3.6))
    if len(factors) == 1:
        axes = [axes]

    for ax, factor in zip(axes, factors):
        labels = [str(v) for v in SWEEPS[factor]]
        means, stds = [], []
        for value in SWEEPS[factor]:
            errs = [r["max_err"] for r in rows
                    if r["factor"] == factor and str(r["value"]) == str(value)]
            means.append(np.mean(errs) if errs else np.nan)
            stds.append(np.std(errs) if errs else np.nan)
        x = np.arange(len(labels))
        ax.errorbar(x, means, yerr=stds, marker="o", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel(factor)
        ax.set_ylabel("max abs error")
        ax.set_title(f"Effect of {factor}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
