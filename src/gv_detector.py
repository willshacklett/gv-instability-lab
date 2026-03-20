"""
gv_detector.py

Core GV detector for early instability detection and simple collapse-vs-recovery
classification from a 1D time series.

What it does:
- Smooths the signal
- Computes first derivative, second derivative, rolling variance
- Builds a "constraint potential" from curvature + variance + slope
- Accumulates potential into a pressure trace
- Triggers when sustained pressure crosses a threshold
- Classifies the event as:
    - "destabilizing"
    - "recovering"
    - "no_flag"

This is intentionally simple, reproducible, and easy to tune.

Example:
    detector = GVDetector()
    result = detector.detect(series)

    print(result.flagged)
    print(result.predicted_index)
    print(result.classification)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class GVResult:
    """Container for detector output."""

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
    """
    Constraint-based instability detector.

    Core idea:
    - instability tends to show up as a combination of:
        1) increasing variance
        2) increasing slope
        3) increasing curvature
        4) persistence (not just a one-off spike)

    We build a local "potential" signal and then accumulate it to produce a
    threshold-crossing event.

    Tunable parameters are exposed in __init__.
    """

    def __init__(
        self,
        smooth_window: int = 5,
        variance_window: int = 8,
        alpha_var: float = 0.25,
        alpha_slope: float = 0.10,
        alpha_curve: float = 0.45,
        cumulative_decay: float = 0.92,
        threshold: float = 25.0,
        min_persistence: int = 3,
        recovery_lookahead: int = 10,
        recovery_drop_ratio: float = 0.35,
        eps: float = 1e-9,
    ) -> None:
        """
        Parameters
        ----------
        smooth_window:
            Moving average window for denoising the raw series.
        variance_window:
            Rolling window used for local variance estimation.
        alpha_var:
            Weight controlling exponential amplification of local variance.
        alpha_slope:
            Weight for positive slope contribution.
        alpha_curve:
            Weight for positive curvature contribution.
        cumulative_decay:
            Memory factor for accumulated potential. Lower means faster decay.
        threshold:
            Trigger threshold for cumulative pressure.
        min_persistence:
            Number of consecutive above-local-baseline points required before
            confirming a trigger.
        recovery_lookahead:
            Number of steps after trigger to inspect for recovery vs collapse.
        recovery_drop_ratio:
            If cumulative pressure drops by this fraction after trigger, classify
            as recovering instead of destabilizing.
        eps:
            Small number for numerical safety.
        """
        if smooth_window < 1:
            raise ValueError("smooth_window must be >= 1")
        if variance_window < 2:
            raise ValueError("variance_window must be >= 2")
        if not (0.0 <= cumulative_decay <= 1.0):
            raise ValueError("cumulative_decay must be between 0 and 1")
        if min_persistence < 1:
            raise ValueError("min_persistence must be >= 1")
        if recovery_lookahead < 1:
            raise ValueError("recovery_lookahead must be >= 1")
        if threshold <= 0:
            raise ValueError("threshold must be > 0")

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
        """
        Run GV detection on a 1D series.

        Parameters
        ----------
        series:
            Numeric 1D sequence.

        Returns
        -------
        GVResult
        """
        x = self._validate_series(series)
        smoothed = self._moving_average(x, self.smooth_window)
        slope = self._gradient(smoothed)
        curvature = self._gradient(slope)
        variance = self._rolling_variance(smoothed, self.variance_window)

        potential = self._compute_potential(
            variance=variance,
            slope=slope,
            curvature=curvature,
        )
        cumulative = self._accumulate(potential)

        predicted_index = self._find_trigger_index(
            cumulative_trace=cumulative,
            potential_trace=potential,
        )

        flagged = predicted_index is not None
        classification = "no_flag"
        trigger_value: Optional[float] = None

        if flagged and predicted_index is not None:
            trigger_value = float(cumulative[predicted_index])
            classification = self._classify_post_trigger(
                cumulative_trace=cumulative,
                trigger_index=predicted_index,
            )

        metadata = {
            "length": int(len(x)),
            "smooth_window": self.smooth_window,
            "variance_window": self.variance_window,
            "alpha_var": self.alpha_var,
            "alpha_slope": self.alpha_slope,
            "alpha_curve": self.alpha_curve,
            "cumulative_decay": self.cumulative_decay,
            "threshold": self.threshold,
            "min_persistence": self.min_persistence,
            "recovery_lookahead": self.recovery_lookahead,
            "recovery_drop_ratio": self.recovery_drop_ratio,
        }

        return GVResult(
            flagged=flagged,
            predicted_index=predicted_index,
            classification=classification,
            threshold=float(self.threshold),
            trigger_value=trigger_value,
            potential_trace=potential.tolist(),
            cumulative_trace=cumulative.tolist(),
            smoothed_series=smoothed.tolist(),
            slope_trace=slope.tolist(),
            curvature_trace=curvature.tolist(),
            variance_trace=variance.tolist(),
            metadata=metadata,
        )

    def score(self, series: List[float] | np.ndarray) -> Dict[str, Any]:
        """
        Convenience wrapper returning dict output.
        """
        return self.detect(series).to_dict()

    def _validate_series(self, series: List[float] | np.ndarray) -> np.ndarray:
        x = np.asarray(series, dtype=float).reshape(-1)
        if x.ndim != 1:
            raise ValueError("series must be 1D")
        if len(x) < max(self.smooth_window, self.variance_window) + 5:
            raise ValueError(
                "series is too short for configured windows; "
                "please provide a longer series"
            )
        if np.isnan(x).any() or np.isinf(x).any():
            raise ValueError("series contains NaN or inf values")
        return x

    def _moving_average(self, x: np.ndarray, window: int) -> np.ndarray:
        if window == 1:
            return x.copy()

        kernel = np.ones(window, dtype=float) / window
        padded = np.pad(x, (window // 2, window - 1 - window // 2), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    def _gradient(self, x: np.ndarray) -> np.ndarray:
        return np.gradient(x)

    def _rolling_variance(self, x: np.ndarray, window: int) -> np.ndarray:
        out = np.zeros_like(x)
        half = window // 2

        for i in range(len(x)):
            left = max(0, i - half)
            right = min(len(x), i + half + 1)
            out[i] = np.var(x[left:right])

        return out

    def _robust_z(self, x: np.ndarray) -> np.ndarray:
        median = np.median(x)
        mad = np.median(np.abs(x - median)) + self.eps
        return (x - median) / (1.4826 * mad + self.eps)

    def _compute_potential(
        self,
        variance: np.ndarray,
        slope: np.ndarray,
        curvature: np.ndarray,
    ) -> np.ndarray:
        """
        Build local potential from variance, slope, and curvature.

        Design choices:
        - Variance enters exponentially so that persistent disorder ramps fast.
        - Only positive slope contributes to destabilization.
        - Only positive curvature contributes strongly to pre-instability buildup.
        """
        z_var = self._robust_z(variance)
        z_slope = self._robust_z(slope)
        z_curve = self._robust_z(curvature)

        pos_slope = np.maximum(z_slope, 0.0)
        pos_curve = np.maximum(z_curve, 0.0)

        # Exponential variance amplification
        var_term = np.exp(self.alpha_var * np.maximum(z_var, 0.0))

        # Add weighted geometry terms
        raw_potential = (
            var_term
            + self.alpha_slope * pos_slope
            + self.alpha_curve * pos_curve
        )

        # Remove tiny values so background noise does not accumulate endlessly
        raw_potential = np.maximum(raw_potential - 1.0, 0.0)

        return raw_potential

    def _accumulate(self, potential: np.ndarray) -> np.ndarray:
        """
        Accumulate potential with memory/decay.
        """
        cumulative = np.zeros_like(potential)
        running = 0.0

        for i, p in enumerate(potential):
            running = self.cumulative_decay * running + p
            cumulative[i] = running

        return cumulative

    def _find_trigger_index(
        self,
        cumulative_trace: np.ndarray,
        potential_trace: np.ndarray,
    ) -> Optional[int]:
        """
        Trigger when:
        - cumulative trace crosses threshold
        - and the local potential has persisted enough to not be a single blip
        """
        above_threshold = cumulative_trace >= self.threshold
        if not np.any(above_threshold):
            return None

        local_baseline = np.median(potential_trace) + np.std(potential_trace)

        for i in range(len(cumulative_trace)):
            if not above_threshold[i]:
                continue

            start = max(0, i - self.min_persistence + 1)
            recent = potential_trace[start : i + 1]
            if len(recent) < self.min_persistence:
                continue

            if np.all(recent > local_baseline):
                return int(i)

        # fallback: first threshold crossing if persistence filter is too strict
        crossings = np.where(above_threshold)[0]
        return int(crossings[0]) if len(crossings) else None

    def _classify_post_trigger(
        self,
        cumulative_trace: np.ndarray,
        trigger_index: int,
    ) -> str:
        """
        Decide whether the triggered event is destabilizing or recovering.

        If post-trigger cumulative pressure decays enough within the lookahead
        window, call it recovering. Otherwise call it destabilizing.
        """
        trigger_val = cumulative_trace[trigger_index]
        if trigger_val <= self.eps:
            return "no_flag"

        end = min(len(cumulative_trace), trigger_index + self.recovery_lookahead + 1)
        post = cumulative_trace[trigger_index:end]

        if len(post) <= 1:
            return "destabilizing"

        post_peak = float(np.max(post))
        post_min = float(np.min(post[1:])) if len(post) > 1 else float(post_peak)

        # If it meaningfully decays after trigger, treat as recovery
        drop = (post_peak - post_min) / (post_peak + self.eps)
        if drop >= self.recovery_drop_ratio:
            return "recovering"

        return "destabilizing"


def detect_gv(
    series: List[float] | np.ndarray,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Functional convenience API.

    Example:
        result = detect_gv(series, threshold=22.0)
    """
    detector = GVDetector(**kwargs)
    return detector.score(series)


if __name__ == "__main__":
    # Tiny smoke test
    toy_series = np.array(
        [
            0.85, 0.88, 0.91, 0.95, 1.02, 1.06, 1.11, 1.20,
            1.35, 1.42, 1.60, 1.85, 2.20, 2.75, 3.20, 3.65,
            4.25, 5.10, 6.35, 8.20, 10.80, 14.50, 19.70, 26.40,
            35.10, 46.20, 60.50
        ],
        dtype=float,
    )

    detector = GVDetector()
    result = detector.detect(toy_series)

    print("Flagged:", result.flagged)
    print("Predicted index:", result.predicted_index)
    print("Classification:", result.classification)
    print("Trigger value:", result.trigger_value)
