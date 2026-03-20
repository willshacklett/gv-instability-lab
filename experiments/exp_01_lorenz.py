"""
exp_01_lorenz.py

Synthetic experiment using Lorenz system to test instability detection.

What this does:
- Generates Lorenz attractor time series
- Extracts one dimension (z) as observable
- Adds optional noise
- Runs GV detector
- Runs baseline detectors
- Prints comparison
- (Optional) plots results

Run:
    python experiments/exp_01_lorenz.py
"""

import numpy as np
import matplotlib.pyplot as plt

from src.gv_detector import GVDetector
from src.baseline_detectors import run_all_baselines


# ----------------------------
# Lorenz System Generator
# ----------------------------

def generate_lorenz(
    n_steps=200,
    dt=0.01,
    sigma=10.0,
    rho=28.0,
    beta=8.0 / 3.0,
    noise_std=0.0,
):
    xs = np.zeros(n_steps)
    ys = np.zeros(n_steps)
    zs = np.zeros(n_steps)

    # initial condition
    xs[0], ys[0], zs[0] = (1.0, 1.0, 1.0)

    for i in range(1, n_steps):
        x, y, z = xs[i - 1], ys[i - 1], zs[i - 1]

        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z

        xs[i] = x + dx * dt
        ys[i] = y + dy * dt
        zs[i] = z + dz * dt

    if noise_std > 0:
        zs = zs + np.random.normal(0, noise_std, size=len(zs))

    return xs, ys, zs


# ----------------------------
# Run Experiment
# ----------------------------

def run_experiment():
    print("\n=== GV Instability Test: Lorenz ===\n")

    # generate data
    _, _, z = generate_lorenz(
        n_steps=120,
        noise_std=0.5,  # simulate noisy observation
    )

    series = z.tolist()

    # ----------------------------
    # Run GV
    # ----------------------------
    gv = GVDetector(
        threshold=25,
        smooth_window=5,
        variance_window=8,
    )

    gv_result = gv.detect(series)

    # ----------------------------
    # Run Baselines
    # ----------------------------
    baseline_results = run_all_baselines(series)

    # ----------------------------
    # Print Results
    # ----------------------------
    print("GV RESULT")
    print("------------------")
    print("Flagged:", gv_result.flagged)
    print("Index:", gv_result.predicted_index)
    print("Class:", gv_result.classification)
    print()

    print("BASELINES")
    print("------------------")
    for name, res in baseline_results.items():
        print(f"{name.upper():10} -> {res['predicted_index']}")

    # ----------------------------
    # Scoreboard (clean view)
    # ----------------------------
    print("\nSCOREBOARD")
    print("------------------")
    print(f"GV         : {gv_result.predicted_index}")
    for name, res in baseline_results.items():
        print(f"{name:<10}: {res['predicted_index']}")

    # ----------------------------
    # Plot
    # ----------------------------
    plot_results(
        series,
        gv_result,
        baseline_results,
    )


# ----------------------------
# Plotting
# ----------------------------

def plot_results(series, gv_result, baseline_results):
    x = np.arange(len(series))

    plt.figure(figsize=(12, 6))

    # main signal
    plt.plot(x, series, label="Signal", linewidth=2)

    # GV trigger
    if gv_result.predicted_index is not None:
        plt.axvline(
            gv_result.predicted_index,
            linestyle="--",
            label="GV",
        )

    # baseline triggers
    for name, res in baseline_results.items():
        if res["predicted_index"] is not None:
            plt.axvline(
                res["predicted_index"],
                linestyle=":",
                label=name,
            )

    plt.title("GV vs Baselines (Lorenz Signal)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ----------------------------
# Entry
# ----------------------------

if __name__ == "__main__":
    run_experiment()
