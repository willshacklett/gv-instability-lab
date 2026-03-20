"""
exp_02_counterfactual.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.gv_detector import GVDetector
from src.baseline_detectors import run_all_baselines


def generate_counterfactual_pair(n_steps=140, noise_std=0.18, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps, dtype=float)

    # Shared early buildup
    shared = 0.015 * t + 0.0009 * (t ** 2) + 0.20 * np.sin(0.22 * t)

    pressure = np.zeros(n_steps)
    half = n_steps // 2
    pressure[:half] = np.linspace(0, 2.8, half)

    early = shared + pressure

    # -----------------------------
    # FAILURE BRANCH (pure escalation)
    # -----------------------------
    failure = early.copy()
    post = np.arange(n_steps - half, dtype=float)

    failure[half:] = (
        failure[half - 1]
        + 0.25 * post
        + 0.030 * (post ** 2)
        + 0.0025 * (post ** 3)
    )

    # -----------------------------
    # RECOVERY BRANCH (relaxation)
    # -----------------------------
    recovery = early.copy()
    recovery[half:] = (
        recovery[half - 1]
        + 0.12 * post
        - 0.010 * (post ** 2)
        + 0.00006 * (post ** 3)
        + 0.18 * np.sin(0.22 * post)
    )

    # Add noise
    failure += rng.normal(0, noise_std, n_steps)
    recovery += rng.normal(0, noise_std, n_steps)

    return failure, recovery


def run_one_case(name, series):
    detector = GVDetector()
    gv = detector.detect(series)
    baselines = run_all_baselines(series)

    print(f"\n=== {name.upper()} ===")
    print("GV")
    print("------------------")
    print("flagged          :", gv.flagged)
    print("predicted_index  :", gv.predicted_index)
    print("classification   :", gv.classification)
    print("trigger_value    :", gv.trigger_value)

    print("\nBaselines")
    print("------------------")
    for k, v in baselines.items():
        print(f"{k:10} -> {v['predicted_index']}")

    return gv


def run():
    print("\n=== COUNTERFACTUAL TEST ===")

    failure, recovery = generate_counterfactual_pair()

    gv_fail = run_one_case("failure_system", failure)
    gv_rec = run_one_case("recovery_system", recovery)

    print("\n\nINTERPRETATION")
    print("======================")

    if gv_fail.flagged and gv_fail.classification == "destabilizing":
        print("GV correctly classified failure as destabilizing")
    else:
        print("GV did NOT clearly classify failure as destabilizing")

    if (not gv_rec.flagged) or (gv_rec.classification == "recovering"):
        print("GV correctly relaxes on recovery")
    else:
        print("GV may be over-triggering recovery")


if __name__ == "__main__":
    run()
