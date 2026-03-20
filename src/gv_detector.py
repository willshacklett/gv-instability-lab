"""
gv_detector.py
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class GVResult:
    flagged: bool
    predicted_index: Optional[int]
    classification: str
    threshold: float
    trigger_value: Optional[float]
    potential_trace: List[float]
    cumulative_trace: List[float]
    smoothed_series: List[float]
    slope_trace: List[float]
    curvature_trace: List[float]
    variance_trace: List[float]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GVDetector:
    def __init__(
        self,
        smooth_window: int = 5,
        variance_window: int = 8,
        alpha_var: float = 0.45,
        alpha_slope: float = 0.18,
        alpha_curve: float = 0.85,
        cumulative_decay: float = 0.97,
        threshold: float = 12.0,
        min_persistence: int = 2,
        recovery_lookahead: int = 12,
        recovery_drop_ratio: float = 0.22,
        eps: float = 1e-9,
    ) -> None:
        self.smooth_window = smooth_window
        self.variance_window = variance_window
        self.alpha_var = alpha_var
        self.alpha_slope = alpha_slope
        self.alpha_curve = alpha_curve
        self.cumulative_decay = cumulative_decay
        self.threshold = threshold
        self.min_persistence = min_persistence
        self.recovery_lookahead = recovery_lookahead
        self.recovery_drop_ratio = recovery_drop_ratio
        self.eps = eps

    def detect(self, series: List[float] | np.ndarray) -> GVResult:
        x = self._validate_series(series)

        smoothed = self._moving_average(x, self.smooth_window)
        slope = np.gradient(smoothed)
        curvature = np.gradient(slope)
        variance = self._rolling_variance(smoothed, self.variance_window)

        potential = self._compute_potential(variance, slope, curvature)
        cumulative = self._accumulate(potential)

        predicted_index = self._find_trigger_index(cumulative, potential)

        flagged = predicted_index is not None
        classification = "no_flag"
        trigger_value = None

        if flagged:
            trigger_value = float(cumulative[predicted_index])
            classification = self._classify_post_trigger(cumulative, predicted_index)

        return GVResult(
            flagged=flagged,
            predicted_index=predicted_index,
            classification=classification,
            threshold=self.threshold,
            trigger_value=trigger_value,
            potential_trace=potential.tolist(),
            cumulative_trace=cumulative.tolist(),
            smoothed_series=smoothed.tolist(),
            slope_trace=slope.tolist(),
            curvature_trace=curvature.tolist(),
            variance_trace=variance.tolist(),
            metadata={},
        )

    def _validate_series(self, series):
        x = np.asarray(series, dtype=float).reshape(-1)
        if len(x) < 20:
            raise ValueError("series too short")
        return x

    def _moving_average(self, x, window):
        if window == 1:
            return x
        kernel = np.ones(window) / window
        padded = np.pad(x, (window // 2, window - 1 - window // 2), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    def _rolling_variance(self, x, window):
        out = np.zeros_like(x)
        half = window // 2
        for i in range(len(x)):
            left = max(0, i - half)
            right = min(len(x), i + half + 1)
            out[i] = np.var(x[left:right])
        return out

    def _robust_z(self, x):
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + self.eps
        return (x - med) / (1.4826 * mad + self.eps)

    def _compute_potential(self, variance, slope, curvature):
        z_var = self._robust_z(variance)
        z_slope = self._robust_z(slope)
        z_curve = self._robust_z(curvature)

        pos_var = np.maximum(z_var, 0.0)
        pos_slope = np.maximum(z_slope, 0.0)
        pos_curve = np.maximum(z_curve, 0.0)

        var_term = np.exp(self.alpha_var * pos_var)

        raw = (
            var_term
            + self.alpha_slope * pos_slope
            + self.alpha_curve * pos_curve
        )

        return np.maximum(raw - 1.0, 0.0)

    def _accumulate(self, potential):
        out = np.zeros_like(potential)
        running = 0.0
        for i, p in enumerate(potential):
            running = self.cumulative_decay * running + p
            out[i] = running
        return out

    def _find_trigger_index(self, cumulative, potential):
        above = cumulative >= self.threshold
        if not np.any(above):
            return None

        baseline = np.median(potential) + 0.75 * np.std(potential)

        for i in range(len(cumulative)):
            if not above[i]:
                continue

            recent = potential[max(0, i - self.min_persistence + 1): i + 1]
            if len(recent) >= self.min_persistence and np.all(recent > baseline):
                return i

        return int(np.where(above)[0][0])

    def _classify_post_trigger(self, cumulative, idx):
        trigger_val = cumulative[idx]
        end = min(len(cumulative), idx + self.recovery_lookahead)
        post = cumulative[idx:end]

        if len(post) < 3:
            return "destabilizing"

        post_peak = float(np.max(post))
        post_last = float(post[-1])
        post_min = float(np.min(post[1:]))

        drop = (post_peak - post_min) / (post_peak + self.eps)

        # true recovery requires both a meaningful drop and a clearly lower ending state
        if drop >= self.recovery_drop_ratio and post_last < trigger_val * 0.80:
            return "recovering"

        # anything still elevated or rising counts as destabilizing
        if post_last >= trigger_val * 0.90:
            return "destabilizing"

        if post_peak >= trigger_val * 1.05:
            return "destabilizing"

        return "recovering"


def detect_gv(series, **kwargs):
    return GVDetector(**kwargs).detect(series).to_dict()


if __name__ == "__main__":
    s = np.linspace(1, 50, 50) ** 1.2
    d = GVDetector()
    r = d.detect(s)
    print(r.classification, r.predicted_index)
