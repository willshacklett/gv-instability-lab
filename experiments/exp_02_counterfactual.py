"""
exp_02_counterfactual.py

Counterfactual instability experiment.

Goal:
- Build two paired systems with similar early buildup
- One system destabilizes
- One system recovers / stabilizes
- Run GV and baselines on both
- Compare whether GV can distinguish collapse vs recovery

Run:
    python experiments/exp_02_counterfactual.py
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt

# Allow running from repo root:
#   python experiments/exp_02_counterfactual.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.gv_detector import GVDetector
from src.baseline_detectors import run_all_baselines


def generate_counterfactual_pair(
    n_steps: int = 140,
    noise_std: float = 0.18,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate two paired series:
    - failure_series: early buildup continues into collapse/divergence
    - recovery_series: similar early buildup but later relaxes and stabilizes

    Design:
    - shared early segment
    - branch after midpoint
    """
    rng = np.random.default_rng(seed)

    t = np.arange(n_steps, dtype=float)

    # Shared early buildup: gentle curve + oscillation + noise
    shared = (
        0.015 * t
        + 0.0009 * (t ** 2)
        + 0.20 * np.sin(0.22 * t)
    )

    # Add modest increasing pressure in the first half
    pressure = np.zeros(n_steps, dtype=float)
    half = n_steps // 2
    pressure[:half] = np.linspace(0.0, 2.8, half)

    early = shared + pressure

    # Branch A: continued divergence / collapse
    failure = early.copy()
    post_t = np.arange(n_steps - half, dtype=float)
    failure[half:] = (
        failure[half - 1]
        + 0.10 * post_t
        + 0.010 * (post_t ** 2)
        + 0.0009 * (post_t ** 3)
        + 0.25 * np.sin(0.30 * post_t)
    )

    # Branch B: buildup appears similar, then dissipates / stabilizes
    recovery = early.copy()
    recovery[half:] = (
        recovery[half - 1]
        + 0.13 * post_t
        - 0.0075 * (post_t ** 2)
        + 0.00008 * (post_t ** 3)
        + 0.22 * np.sin(0.28 * post_t)
    )

    # Add observational noise
    failure += rng.normal(0.0, noise_std, size=n_steps)
    recovery += rng.normal(0.0, noise_std, size=n_steps)

    return failure, recovery


def infer_actual_transition_index(
    series: np.ndarray,
    mode: str,
    smooth_window: int = 7,
) -> int | None:
    """
    Rough reference label for plotting/reporting only.

    For failure:
    - detect when curvature + slope jointly become persistently positive
      and the tail continues upward strongly

    For recovery:
    - return None, since we treat it as "no true collapse"

    This is not used by GV itself; it is only a reporting helper.
    """
    if mode == "recovery":
        return None

    smoothed = moving_average(series, smooth_window)
    slope = np.gradient(smoothed)
    curvature = np.gradient(slope)

    pos_slope = slope > np.percentile(slope, 70)
    pos_curve = curvature > np.percentile(curvature, 70)

    for i in range(8, len(series) - 8):
        if np.all(pos_slope[i - 2:i + 2]) and np.all(pos_curve[i - 2:i + 2]):
            return i

    return int(np.argmax(smoothed))


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(x, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def run_one_case(
    name: str,
    series: np.ndarray,
    actual_transition: int | None,
) -> Dict[str, Any]:
    """
    Run GV and baselines on a single series and print results.
    """
    detector = GVDetector(
        smooth_window=5,
        variance_window=8,
        alpha_var=0.25,
        alpha_slope=0.10,
        alpha_curve=0.45,
        cumulative_decay=0.92,
        threshold=10.5,
        min_persistence=3,
        recovery_lookahead=14,
        recovery_drop_ratio=0.28,
    )

    gv_result = detector.detect(series)
    baselines = run_all_baselines(series)

    print(f"\n=== {name.upper()} ===")
    print("GV")
    print("---------------")
    print("flagged         :", gv_result.flagged)
    print("predicted_index :", gv_result.predicted_index)
    print("classification  :", gv_result.classification)
    print("trigger_value   :", gv_result.trigger_value)
    print("actual_transition:", actual_transition)

    print("\nBaselines")
    print("---------------")
    for method, res in baselines.items():
        print(f"{method:10} -> {res['predicted_index']}")

    return {
        "gv": gv_result,
        "baselines": baselines,
        "actual_transition": actual_transition,
    }


def plot_case(
    ax: plt.Axes,
    title: str,
    series: np.ndarray,
    gv_result,
    baselines: Dict[str, Dict[str, Any]],
    actual_transition: int | None,
) -> None:
    x = np.arange(len(series))

    ax.plot(x, series, linewidth=2, label="signal")

    if actual_transition is not None:
        ax.axvline(
            actual_transition,
            linestyle="-.",
            linewidth=2,
            label="actual_transition",
        )

    if gv_result.predicted_index is not None:
        ax.axvline(
            gv_result.predicted_index,
            linestyle="--",
            linewidth=2,
            label=f"GV ({gv_result.classification})",
        )

    for method, res in baselines.items():
        if res["predicted_index"] is not None:
            ax.axvline(
                res["predicted_index"],
                linestyle=":",
                alpha=0.85,
                label=method,
            )

    ax.set_title(title)
    ax.legend(fontsize=8)


def summarize_counterfactual(
    failure_result: Dict[str, Any],
    recovery_result: Dict[str, Any],
) -> None:
    """
    Print a compact summary focused on the question:
    can GV distinguish collapse vs recovery?
    """
    f_gv = failure_result["gv"]
    r_gv = recovery_result["gv"]

    print("\n\nCOUNTERFACTUAL SUMMARY")
    print("======================")
    print("Failure series")
    print(f"  GV flagged        : {f_gv.flagged}")
    print(f"  GV index          : {f_gv.predicted_index}")
    print(f"  GV classification : {f_gv.classification}")

    print("Recovery series")
    print(f"  GV flagged        : {r_gv.flagged}")
    print(f"  GV index          : {r_gv.predicted_index}")
    print(f"  GV classification : {r_gv.classification}")

    print("\nInterpretation")
    if f_gv.flagged and f_gv.classification == "destabilizing":
        print("  - GV identifies genuine buildup leading toward collapse.")
    else:
        print("  - GV did NOT clearly classify the failure case as destabilizing.")

    if (not r_gv.flagged) or (r_gv.classification == "recovering"):
        print("  - GV recognizes or relaxes on the recovery case.")
    else:
        print("  - GV may be over-triggering on recovery.")

    print("\nWhat to look for")
    print("  - Failure should usually flag earlier than baselines.")
    print("  - Recovery should avoid a hard destabilizing classification.")
    print("  - If recovery is flagged, it should ideally classify as recovering.")


def run_experiment() -> None:
    print("\n=== Counterfactual Experiment: Collapse vs Recovery ===")

    failure_series, recovery_series = generate_counterfactual_pair(
        n_steps=140,
        noise_std=0.18,
        seed=42,
    )

    actual_failure_transition = infer_actual_transition_index(
        failure_series,
        mode="failure",
    )
    actual_recovery_transition = infer_actual_transition_index(
        recovery_series,
        mode="recovery",
    )

    failure_result = run_one_case(
        "failure_system",
        failure_series,
        actual_failure_transition,
    )
    recovery_result = run_one_case(
        "recovery_system",
        recovery_series,
        actual_recovery_transition,
    )

    summarize_counterfactual(failure_result, recovery_result)

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    plot_case(
        axes[0],
        "System A: Failure / Destabilization",
        failure_series,
        failure_result["gv"],
        failure_result["baselines"],
        failure_result["actual_transition"],
    )

    plot_case(
        axes[1],
        "System B: Recovery / Stabilization",
        recovery_series,
        recovery_result["gv"],
        recovery_result["baselines"],
        recovery_result["actual_transition"],
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_experiment()
