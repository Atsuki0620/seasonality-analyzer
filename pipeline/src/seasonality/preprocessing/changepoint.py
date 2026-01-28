"""Changepoint detection utilities for time series data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import ruptures as rpt
from loguru import logger


@dataclass
class Segment:
    """A segment of time series data between changepoints."""

    start_idx: pd.Timestamp
    end_idx: pd.Timestamp
    segment_number: int
    length: int
    data: pd.Series


def detect_changepoints_pelt(
    series: pd.Series,
    model: Literal["rbf", "l1", "l2", "normal"] = "rbf",
    min_size: int = 30,
    penalty: float = 10.0,
) -> List[pd.Timestamp]:
    """Detect changepoints using PELT algorithm.

    PELT (Pruned Exact Linear Time) is an exact algorithm for
    detecting multiple changepoints with linear time complexity.

    Args:
        series: Input time series (must be without NaN).
        model: Cost model ('rbf', 'l1', 'l2', 'normal').
        min_size: Minimum segment size.
        penalty: Penalty parameter controlling number of changepoints.

    Returns:
        List of changepoint timestamps.
    """
    data = series.dropna()
    if len(data) < 2 * min_size:
        logger.warning(f"Data too short for changepoint detection: {len(data)} < {2 * min_size}")
        return []

    values = data.values

    algo = rpt.Pelt(model=model, min_size=min_size).fit(values)
    result = algo.predict(pen=penalty)

    # Convert indices to timestamps (excluding the last element which is len(data))
    change_indices = [data.index[i - 1] for i in result[:-1] if i < len(data)]

    logger.debug(f"PELT detected {len(change_indices)} changepoints")
    return change_indices


def detect_changepoints_window(
    series: pd.Series,
    model: Literal["rbf", "l1", "l2", "normal"] = "rbf",
    min_size: int = 30,
    width: int = 50,
    penalty: float = 10.0,
) -> List[pd.Timestamp]:
    """Detect changepoints using sliding window algorithm.

    Window-based detection is more suitable for local changes.

    Args:
        series: Input time series (must be without NaN).
        model: Cost model.
        min_size: Minimum segment size.
        width: Window width.
        penalty: Penalty parameter.

    Returns:
        List of changepoint timestamps.
    """
    data = series.dropna()
    if len(data) < 2 * min_size:
        logger.warning(f"Data too short for changepoint detection: {len(data)} < {2 * min_size}")
        return []

    values = data.values

    algo = rpt.Window(model=model, min_size=min_size, width=width).fit(values)
    result = algo.predict(pen=penalty)

    # Convert indices to timestamps
    change_indices = [data.index[i - 1] for i in result[:-1] if i < len(data)]

    logger.debug(f"Window detected {len(change_indices)} changepoints")
    return change_indices


def detect_changepoints(
    series: pd.Series,
    method: Literal["pelt", "window"] = "pelt",
    model: Literal["rbf", "l1", "l2", "normal"] = "rbf",
    min_size: int = 30,
    penalty: float = 10.0,
    width: int = 50,
) -> List[pd.Timestamp]:
    """Detect changepoints using specified method.

    Args:
        series: Input time series.
        method: Detection method ('pelt' or 'window').
        model: Cost model for ruptures.
        min_size: Minimum segment size.
        penalty: Penalty parameter.
        width: Window width (for window method).

    Returns:
        List of changepoint timestamps.
    """
    if method == "pelt":
        return detect_changepoints_pelt(
            series, model=model, min_size=min_size, penalty=penalty
        )
    elif method == "window":
        return detect_changepoints_window(
            series, model=model, min_size=min_size, width=width, penalty=penalty
        )
    else:
        raise ValueError(f"Unknown changepoint method: {method}")


def segment_by_changepoints(
    series: pd.Series,
    changepoints: List[pd.Timestamp],
) -> List[Segment]:
    """Split time series into segments at changepoints.

    Args:
        series: Input time series.
        changepoints: List of changepoint timestamps.

    Returns:
        List of Segment objects.
    """
    segments = []
    boundaries = [series.index[0]] + sorted(changepoints) + [series.index[-1]]

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]

        # Get segment data (inclusive on both ends)
        mask = (series.index >= start) & (series.index <= end)
        segment_data = series[mask]

        segments.append(Segment(
            start_idx=start,
            end_idx=end,
            segment_number=i + 1,
            length=len(segment_data),
            data=segment_data,
        ))

    return segments


def segment_statistics(
    series: pd.Series,
    changepoints: List[pd.Timestamp],
) -> pd.DataFrame:
    """Calculate statistics for each segment.

    Args:
        series: Input time series.
        changepoints: List of changepoint timestamps.

    Returns:
        DataFrame with segment statistics.
    """
    segments = segment_by_changepoints(series, changepoints)

    stats = []
    for seg in segments:
        stats.append({
            "segment": seg.segment_number,
            "start": seg.start_idx,
            "end": seg.end_idx,
            "n_points": seg.length,
            "mean": seg.data.mean(),
            "std": seg.data.std(),
            "min": seg.data.min(),
            "max": seg.data.max(),
            "median": seg.data.median(),
        })

    return pd.DataFrame(stats)
