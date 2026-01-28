"""Missing value imputation utilities for time series data."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd
from loguru import logger


def interpolate_missing(
    series: pd.Series,
    method: Literal["linear", "spline", "polynomial", "time"] = "linear",
    limit: Optional[int] = None,
    limit_direction: Literal["forward", "backward", "both"] = "both",
    order: int = 3,
) -> pd.Series:
    """Interpolate missing values in time series.

    Args:
        series: Input time series with timestamp index.
        method: Interpolation method:
            - 'linear': Linear interpolation
            - 'spline': Spline interpolation
            - 'polynomial': Polynomial interpolation
            - 'time': Time-weighted interpolation
        limit: Maximum number of consecutive NaNs to fill.
        limit_direction: Direction to fill ('forward', 'backward', 'both').
        order: Order for spline/polynomial interpolation.

    Returns:
        Series with missing values interpolated.
    """
    if series.isna().sum() == 0:
        return series

    n_missing = series.isna().sum()
    logger.debug(f"Interpolating {n_missing} missing values using {method}")

    if method == "linear":
        return series.interpolate(
            method="linear",
            limit=limit,
            limit_direction=limit_direction,
        )
    elif method == "spline":
        return series.interpolate(
            method="spline",
            order=order,
            limit=limit,
            limit_direction=limit_direction,
        )
    elif method == "polynomial":
        return series.interpolate(
            method="polynomial",
            order=order,
            limit=limit,
            limit_direction=limit_direction,
        )
    elif method == "time":
        return series.interpolate(
            method="time",
            limit=limit,
            limit_direction=limit_direction,
        )
    else:
        raise ValueError(f"Unknown interpolation method: {method}")


def forward_fill(
    series: pd.Series,
    limit: Optional[int] = None,
) -> pd.Series:
    """Forward fill missing values.

    Args:
        series: Input time series.
        limit: Maximum number of consecutive NaNs to fill.

    Returns:
        Series with forward-filled values.
    """
    return series.ffill(limit=limit)


def backward_fill(
    series: pd.Series,
    limit: Optional[int] = None,
) -> pd.Series:
    """Backward fill missing values.

    Args:
        series: Input time series.
        limit: Maximum number of consecutive NaNs to fill.

    Returns:
        Series with backward-filled values.
    """
    return series.bfill(limit=limit)


def fill_with_seasonal_mean(
    series: pd.Series,
    period: int = 7,
) -> pd.Series:
    """Fill missing values with seasonal mean.

    Useful when data has clear seasonality (e.g., day-of-week effects).

    Args:
        series: Input time series with timestamp index.
        period: Seasonal period (default: 7 for weekly).

    Returns:
        Series with seasonally-filled missing values.
    """
    result = series.copy()

    # Calculate day-of-week (or period) means
    if period == 7:
        seasonal_idx = series.index.dayofweek
    else:
        # Use modulo of day count for custom periods
        day_count = (series.index - series.index[0]).days
        seasonal_idx = day_count % period

    # Calculate mean for each seasonal position
    temp_df = pd.DataFrame({"value": series, "season": seasonal_idx})
    seasonal_means = temp_df.groupby("season")["value"].mean()

    # Fill missing values
    for i, val in enumerate(result):
        if pd.isna(val):
            season = seasonal_idx[i] if isinstance(seasonal_idx, pd.Series) else seasonal_idx
            if isinstance(season, (int, np.integer)):
                season_key = season
            else:
                season_key = season.iloc[i] if hasattr(season, "iloc") else season[i]
            if season_key in seasonal_means.index:
                result.iloc[i] = seasonal_means[season_key]

    return result


def impute_missing(
    series: pd.Series,
    method: Literal["linear", "spline", "ffill", "seasonal", "none"] = "linear",
    limit: Optional[int] = None,
    **kwargs,
) -> pd.Series:
    """Impute missing values using specified method.

    Args:
        series: Input time series.
        method: Imputation method:
            - 'linear': Linear interpolation
            - 'spline': Spline interpolation
            - 'ffill': Forward fill
            - 'seasonal': Seasonal mean fill
            - 'none': No imputation
        limit: Maximum consecutive NaNs to fill.
        **kwargs: Additional method-specific parameters.

    Returns:
        Series with imputed values.
    """
    if method == "none":
        return series

    if method == "linear":
        return interpolate_missing(series, method="linear", limit=limit)
    elif method == "spline":
        order = kwargs.get("order", 3)
        return interpolate_missing(series, method="spline", limit=limit, order=order)
    elif method == "ffill":
        return forward_fill(series, limit=limit)
    elif method == "seasonal":
        period = kwargs.get("period", 7)
        return fill_with_seasonal_mean(series, period=period)
    else:
        raise ValueError(f"Unknown imputation method: {method}")


def get_missing_report(series: pd.Series) -> dict:
    """Generate report on missing values in series.

    Args:
        series: Input time series.

    Returns:
        Dictionary with missing value statistics.
    """
    total = len(series)
    missing = series.isna().sum()

    # Find longest gap
    is_missing = series.isna()
    gaps = []
    gap_start = None

    for i, is_na in enumerate(is_missing):
        if is_na and gap_start is None:
            gap_start = i
        elif not is_na and gap_start is not None:
            gaps.append(i - gap_start)
            gap_start = None

    if gap_start is not None:
        gaps.append(len(series) - gap_start)

    return {
        "total_count": total,
        "missing_count": int(missing),
        "missing_rate": missing / total if total > 0 else 0,
        "num_gaps": len(gaps),
        "max_gap": max(gaps) if gaps else 0,
        "mean_gap": sum(gaps) / len(gaps) if gaps else 0,
    }
