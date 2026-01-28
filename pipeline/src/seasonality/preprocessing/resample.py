"""Resampling utilities for time series data."""

from __future__ import annotations

from typing import Literal, Optional

import pandas as pd
from loguru import logger


def resample_to_daily(
    series: pd.Series,
    method: Literal["mean", "median", "ffill"] = "mean",
) -> pd.Series:
    """Resample time series to daily frequency.

    Args:
        series: Input time series with timestamp index.
        method: Aggregation method ('mean', 'median', 'ffill').

    Returns:
        Resampled series with daily frequency.
    """
    if method == "mean":
        return series.resample("D").mean()
    elif method == "median":
        return series.resample("D").median()
    elif method == "ffill":
        return series.resample("D").ffill()
    else:
        raise ValueError(f"Unknown resample method: {method}")


def resample_and_interpolate(
    series: pd.Series,
    resample_freq: str = "D",
    resample_method: Literal["mean", "median", "ffill"] = "mean",
    interp_method: Optional[Literal["linear", "spline", "ffill"]] = "linear",
    spline_order: int = 3,
) -> pd.Series:
    """Resample time series and interpolate missing values.

    Args:
        series: Input time series with timestamp index.
        resample_freq: Resampling frequency (e.g., 'D' for daily).
        resample_method: Aggregation method for resampling.
        interp_method: Interpolation method for missing values.
            None means no interpolation (keep NaN).
        spline_order: Order for spline interpolation.

    Returns:
        Resampled and interpolated series.
    """
    # Resample
    if resample_method == "mean":
        resampled = series.resample(resample_freq).mean()
    elif resample_method == "median":
        resampled = series.resample(resample_freq).median()
    elif resample_method == "ffill":
        resampled = series.resample(resample_freq).ffill()
    else:
        raise ValueError(f"Unknown resample_method: {resample_method}")

    # Interpolate
    if interp_method is None:
        return resampled
    elif interp_method == "linear":
        return resampled.interpolate(method="linear")
    elif interp_method == "spline":
        return resampled.interpolate(method="spline", order=spline_order)
    elif interp_method == "ffill":
        return resampled.ffill()
    else:
        raise ValueError(f"Unknown interp_method: {interp_method}")


def resample_dataframe(
    df: pd.DataFrame,
    sensor_cols: list[str],
    resample_freq: str = "D",
    resample_method: Literal["mean", "median", "ffill"] = "mean",
    interp_method: Optional[Literal["linear", "spline", "ffill"]] = "linear",
) -> pd.DataFrame:
    """Resample entire DataFrame with multiple sensor columns.

    Args:
        df: Input DataFrame with timestamp index.
        sensor_cols: List of sensor column names.
        resample_freq: Resampling frequency.
        resample_method: Aggregation method.
        interp_method: Interpolation method.

    Returns:
        Resampled DataFrame.
    """
    result = pd.DataFrame(index=df.resample(resample_freq).mean().index)

    for col in sensor_cols:
        if col in df.columns:
            result[col] = resample_and_interpolate(
                df[col],
                resample_freq=resample_freq,
                resample_method=resample_method,
                interp_method=interp_method,
            )
        else:
            logger.warning(f"Column {col} not found in DataFrame")

    return result
