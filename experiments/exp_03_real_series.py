"""
exp_03_real_series.py

Run GV and baseline detectors on a real-world time series CSV.

What this does:
- Loads CSV data
- Selects a numeric column
- Optionally normalizes / slices
- Runs GV detector
- Runs baseline detectors
- Plots results

Example CSV format:
    date,close
    2020-01-01,100
    2020-01-02,101
    ...

Run:
    python experiments/exp_03_real_series.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Fix import path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.gv_detector import GVDetector
from src.baseline_detectors import run_all_baselines
from src.datasets import load_series_from_csv


# ----------------------------
# CONFIG (EDIT THIS)
# ----------------------------

DATA_PATH = "data/real/sample.csv"   # <-- change this
VALUE_COLUMN = "value"                  # or "close", "value", etc.

NORMALIZE = True
START_INDEX = None
END_INDEX = None


# ----------------------------
# Load Data
# ----------------------------

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"File not found: {DATA_PATH}\n"
            "Put a CSV in data/real/ and update DATA_PATH."
        )

    series = load_series_from_csv(
        path=DATA_PATH,
        value_column=VALUE_COLUMN,
        normalize=NORMALIZE,
        start_index=START_INDEX,
        end_index=END_INDEX,
    )

    return series


# ----------------------------
# Run Experiment
# ----------------------------

def run_experiment():
    print("\n=== GV Real Data Test ===\n")

    series = load_data()

    print(f"Loaded series length: {len(series)}")

    # ----------------------------
    # Run GV
    # ----------------------------
    gv = GVDetector(
        threshold=10.5,
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
    print("\nGV RESULT")
    print("------------------")
    print("Flagged:", gv_result.flagged)
    print("Index:", gv_result.predicted_index)
    print("Class:", gv_result.classification)

    print("\nBASELINES")
    print("------------------")
    for name, res in baseline_results.items():
        print(f"{name.upper():10} -> {res['predicted_index']}")

    # ----------------------------
    # Plot
    # ----------------------------
    plot_results(series, gv_result, baseline_results)


# ----------------------------
# Plotting
# ----------------------------

def plot_results(series, gv_result, baseline_results):
    x = np.arange(len(series))

    plt.figure(figsize=(12, 6))

    # signal
    plt.plot(x, series, label="Signal", linewidth=2)

    # GV
    if gv_result.predicted_index is not None:
        plt.axvline(
            gv_result.predicted_index,
            linestyle="--",
            label="GV",
        )

    # baselines
    for name, res in baseline_results.items():
        if res["predicted_index"] is not None:
            plt.axvline(
                res["predicted_index"],
                linestyle=":",
                label=name,
            )

    plt.title("GV vs Baselines (Real Data)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ----------------------------
# Entry
# ----------------------------

if __name__ == "__main__":
    run_experiment()
