"""
baseline_detectors.py

Baseline detectors for comparison against GV.

These are intentionally simple and transparent:
- rolling variance threshold
- z-score spike detection
- second derivative (curvature) threshold
- CUSUM (cumulative sum) detector

Each returns:
    {
        "flagged": bool,
        "predicted_index": int or None,
        "score_trace": list,
        "method": str
    }
"""

from typing import Dict, Any, List, Optional

import numpy as np


# ----------------------------
# Utilities
# ----------------------------

def _validate(series: List[float] | np.ndarray) -> np.ndarray:
    x = np.asarray(series, dtype=float).reshape(-1)
    if x.ndim != 1:
        raise ValueError("Series must be 1D")
    if len(x) < 10:
        raise ValueError("Series too short")
    if np.isnan(x).any() or np.isinf(x).any():
        raise ValueError("Series contains NaN or inf")
    return x


def _rolling_variance(x: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros_like(x)
    half = window // 2

    for i in range(len(x)):
        left = max(0, i - half)
        right = min(len(x), i + half + 1)
        out[i] = np.var(x[left:right])

    return out


def _robust_z(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + eps
    return (x - med) / (1.4826 * mad + eps)


# ----------------------------
# 1. Rolling Variance Detector
# ----------------------------

def detect_variance(
    series: List[float] | np.ndarray,
    window: int = 8,
    threshold: float = 2.5,
) -> Dict[str, Any]:
    """
    Flags when rolling variance exceeds threshold (z-scored).
    """
    x = _validate(series)
    var = _rolling_variance(x, window)
    z = _robust_z(var)

    idx = np.where(z > threshold)[0]
    pred = int(idx[0]) if len(idx) else None

    return {
        "flagged": pred is not None,
        "predicted_index": pred,
        "score_trace": z.tolist(),
        "method": "variance_z",
    }


# ----------------------------
# 2. Z-score Spike Detector
# ----------------------------

def detect_zscore(
    series: List[float] | np.ndarray,
    threshold: float = 3.0,
) -> Dict[str, Any]:
    """
    Flags when signal itself spikes relative to robust baseline.
    """
    x = _validate(series)
    z = _robust_z(x)

    idx = np.where(z > threshold)[0]
    pred = int(idx[0]) if len(idx) else None

    return {
        "flagged": pred is not None,
        "predicted_index": pred,
        "score_trace": z.tolist(),
        "method": "zscore",
    }


# ----------------------------
# 3. Curvature (2nd Derivative)
# ----------------------------

def detect_curvature(
    series: List[float] | np.ndarray,
    threshold: float = 2.0,
) -> Dict[str, Any]:
    """
    Flags based on second derivative (acceleration).
    """
    x = _validate(series)
    slope = np.gradient(x)
    curvature = np.gradient(slope)

    z = _robust_z(curvature)

    idx = np.where(z > threshold)[0]
    pred = int(idx[0]) if len(idx) else None

    return {
        "flagged": pred is not None,
        "predicted_index": pred,
        "score_trace": z.tolist(),
        "method": "curvature",
    }


# ----------------------------
# 4. CUSUM Detector
# ----------------------------

def detect_cusum(
    series: List[float] | np.ndarray,
    drift: float = 0.0,
    threshold: float = 5.0,
) -> Dict[str, Any]:
    """
    Cumulative Sum (CUSUM) detector.

    Detects persistent upward drift.
    """
    x = _validate(series)

    mean = np.mean(x)
    s_pos = 0.0
    trace = []

    pred: Optional[int] = None

    for i, val in enumerate(x):
        s_pos = max(0.0, s_pos + (val - mean - drift))
        trace.append(s_pos)

        if s_pos > threshold and pred is None:
            pred = i

    return {
        "flagged": pred is not None,
        "predicted_index": pred,
        "score_trace": trace,
        "method": "cusum",
    }


# ----------------------------
# Run All Baselines
# ----------------------------

def run_all_baselines(
    series: List[float] | np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    """
    Convenience function to run all baseline detectors.
    """
    return {
        "variance": detect_variance(series),
        "zscore": detect_zscore(series),
        "curvature": detect_curvature(series),
        "cusum": detect_cusum(series),
    }


# ----------------------------
# Quick test
# ----------------------------

if __name__ == "__main__":
    toy = np.array([
        0.85, 0.90, 1.05, 1.20, 1.35,
        1.60, 2.00, 2.80, 3.60, 4.80,
        6.20, 8.50, 12.0, 18.0, 27.0
    ])

    results = run_all_baselines(toy)

    for name, res in results.items():
        print(f"\n{name.upper()}")
        print("Flagged:", res["flagged"])
        print("Index:", res["predicted_index"])
