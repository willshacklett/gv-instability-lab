"""
datasets.py

Helpers for loading and preparing 1D time series datasets for GV experiments.

Supports:
- CSV files
- selecting a numeric column
- optional normalization
- optional differencing
- optional slicing

This keeps experiment files clean and makes it easy to swap in real data later.

Example:
    from src.datasets import load_series_from_csv

    series = load_series_from_csv(
        path="data/real/example.csv",
        value_column="close",
        normalize=True,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class SeriesBundle:
    """
    Container for loaded series and metadata.
    """

    values: np.ndarray
    raw_values: np.ndarray
    value_column: str
    source_path: str
    start_index: Optional[int]
    end_index: Optional[int]
    normalized: bool
    differenced: bool

    def to_dict(self) -> dict:
        return {
            "values": self.values.tolist(),
            "raw_values": self.raw_values.tolist(),
            "value_column": self.value_column,
            "source_path": self.source_path,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "normalized": self.normalized,
            "differenced": self.differenced,
        }


def load_csv(
    path: str,
) -> pd.DataFrame:
    """
    Load a CSV file.
    """
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV is empty: {path}")
    return df


def infer_numeric_column(
    df: pd.DataFrame,
    exclude: Optional[Sequence[str]] = None,
) -> str:
    """
    Infer the first usable numeric column.
    """
    exclude_set = set(exclude or [])

    numeric_cols = [
        c for c in df.columns
        if c not in exclude_set and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not numeric_cols:
        raise ValueError("No numeric columns found in CSV")

    return numeric_cols[0]


def zscore_normalize(
    x: np.ndarray,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Standard z-score normalization.
    """
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    return (x - mu) / (sigma + eps)


def minmax_normalize(
    x: np.ndarray,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Min-max normalization to [0, 1].
    """
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    return (x - xmin) / (xmax - xmin + eps)


def difference_series(
    x: np.ndarray,
    order: int = 1,
) -> np.ndarray:
    """
    Difference a series one or more times.
    """
    out = x.copy()
    for _ in range(order):
        out = np.diff(out)
    return out


def slice_series(
    x: np.ndarray,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
) -> np.ndarray:
    """
    Slice a series safely.
    """
    start = 0 if start_index is None else start_index
    end = len(x) if end_index is None else end_index

    if start < 0 or end > len(x) or start >= end:
        raise ValueError(
            f"Invalid slice bounds: start={start}, end={end}, len={len(x)}"
        )

    return x[start:end]


def validate_series(
    x: np.ndarray,
    min_length: int = 20,
) -> np.ndarray:
    """
    Validate a numeric 1D series.
    """
    x = np.asarray(x, dtype=float).reshape(-1)

    if x.ndim != 1:
        raise ValueError("Series must be 1D")
    if len(x) < min_length:
        raise ValueError(
            f"Series too short: length={len(x)}, need at least {min_length}"
        )
    if np.isnan(x).any() or np.isinf(x).any():
        raise ValueError("Series contains NaN or inf values")

    return x


def load_series_from_csv(
    path: str,
    value_column: Optional[str] = None,
    normalize: bool = False,
    normalize_mode: str = "zscore",
    difference: bool = False,
    difference_order: int = 1,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
    dropna: bool = True,
) -> np.ndarray:
    """
    Load a numeric series from CSV and return just the processed values.

    Parameters
    ----------
    path:
        Path to CSV file.
    value_column:
        Column to use. If omitted, first numeric column is inferred.
    normalize:
        Whether to normalize the final series.
    normalize_mode:
        'zscore' or 'minmax'
    difference:
        Whether to difference the series.
    difference_order:
        Number of differencing passes.
    start_index, end_index:
        Optional slicing after load.
    dropna:
        Drop NaNs before processing.

    Returns
    -------
    np.ndarray
    """
    bundle = load_series_bundle_from_csv(
        path=path,
        value_column=value_column,
        normalize=normalize,
        normalize_mode=normalize_mode,
        difference=difference,
        difference_order=difference_order,
        start_index=start_index,
        end_index=end_index,
        dropna=dropna,
    )
    return bundle.values


def load_series_bundle_from_csv(
    path: str,
    value_column: Optional[str] = None,
    normalize: bool = False,
    normalize_mode: str = "zscore",
    difference: bool = False,
    difference_order: int = 1,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
    dropna: bool = True,
) -> SeriesBundle:
    """
    Load a numeric series from CSV and return both processed + raw info.
    """
    df = load_csv(path)

    if value_column is None:
        value_column = infer_numeric_column(df)

    if value_column not in df.columns:
        raise ValueError(
            f"Column '{value_column}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    series = df[value_column]

    if dropna:
        series = series.dropna()

    raw = series.to_numpy(dtype=float)
    raw = validate_series(raw)

    values = slice_series(raw, start_index=start_index, end_index=end_index)

    if difference:
        values = difference_series(values, order=difference_order)

    values = validate_series(values)

    if normalize:
        if normalize_mode == "zscore":
            values = zscore_normalize(values)
        elif normalize_mode == "minmax":
            values = minmax_normalize(values)
        else:
            raise ValueError(
                f"Unknown normalize_mode='{normalize_mode}'. "
                "Use 'zscore' or 'minmax'."
            )

    values = validate_series(values)

    return SeriesBundle(
        values=values,
        raw_values=raw,
        value_column=value_column,
        source_path=path,
        start_index=start_index,
        end_index=end_index,
        normalized=normalize,
        differenced=difference,
    )


def make_train_test_split(
    x: np.ndarray,
    train_fraction: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a series into train/test segments by time.
    """
    x = validate_series(x)

    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be between 0 and 1")

    split_idx = int(len(x) * train_fraction)
    if split_idx < 5 or (len(x) - split_idx) < 5:
        raise ValueError("Train/test split leaves too few points")

    return x[:split_idx], x[split_idx:]


def rolling_windows(
    x: np.ndarray,
    window: int,
    step: int = 1,
) -> list[np.ndarray]:
    """
    Create rolling windows from a 1D series.
    """
    x = validate_series(x)

    if window < 2:
        raise ValueError("window must be >= 2")
    if step < 1:
        raise ValueError("step must be >= 1")
    if window > len(x):
        raise ValueError("window cannot exceed series length")

    out: list[np.ndarray] = []
    for start in range(0, len(x) - window + 1, step):
        out.append(x[start : start + window])

    return out


if __name__ == "__main__":
    # Tiny smoke test with synthetic data
    import tempfile
    from pathlib import Path

    tmpdir = tempfile.mkdtemp()
    csv_path = Path(tmpdir) / "toy.csv"

    df = pd.DataFrame(
        {
            "time": np.arange(40),
            "signal": np.linspace(0.5, 5.0, 40) + 0.2 * np.sin(np.arange(40)),
        }
    )
    df.to_csv(csv_path, index=False)

    bundle = load_series_bundle_from_csv(
        path=str(csv_path),
        value_column="signal",
        normalize=True,
    )

    print("Loaded column:", bundle.value_column)
    print("Length:", len(bundle.values))
    print("First 5 values:", bundle.values[:5].tolist())
