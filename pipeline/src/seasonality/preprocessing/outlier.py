"""Outlier detection utilities for time series data."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from loguru import logger


def detect_outliers_iqr(
    series: pd.Series,
    k: float = 1.5,
) -> pd.Series:
    """Detect outliers using IQR method.

    Args:
        series: Input time series.
        k: Multiplier for IQR (default: 1.5).

    Returns:
        Boolean series indicating outliers.
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR
    return (series < lower) | (series > upper)


def detect_outliers_zscore(
    series: pd.Series,
    threshold: float = 3.0,
) -> pd.Series:
    """Detect outliers using Z-score method.

    Args:
        series: Input time series.
        threshold: Z-score threshold (default: 3.0).

    Returns:
        Boolean series indicating outliers.
    """
    data = series.dropna()
    if len(data) == 0:
        return pd.Series(False, index=series.index)

    z_scores = np.abs(stats.zscore(data))
    outlier_idx = data.index[z_scores > threshold]
    return pd.Series(series.index.isin(outlier_idx), index=series.index)


def detect_outliers_iforest(
    series: pd.Series,
    contamination: float = 0.05,
    random_state: int = 42,
) -> pd.Series:
    """Detect outliers using Isolation Forest.

    Args:
        series: Input time series.
        contamination: Expected proportion of outliers.
        random_state: Random seed for reproducibility.

    Returns:
        Boolean series indicating outliers.
    """
    data = series.dropna()
    if len(data) == 0:
        return pd.Series(False, index=series.index)

    X = data.values.reshape(-1, 1)
    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
    )
    predictions = iso.fit_predict(X)
    outlier_idx = data.index[predictions == -1]
    return pd.Series(series.index.isin(outlier_idx), index=series.index)


def detect_outliers(
    series: pd.Series,
    method: Literal["iqr", "zscore", "iforest"] = "iqr",
    threshold: Optional[float] = None,
    **kwargs,
) -> pd.Series:
    """Detect outliers using specified method.

    Args:
        series: Input time series.
        method: Detection method ('iqr', 'zscore', 'iforest').
        threshold: Method-specific threshold.
        **kwargs: Additional method-specific parameters.

    Returns:
        Boolean series indicating outliers.
    """
    if method == "iqr":
        k = threshold if threshold is not None else 1.5
        return detect_outliers_iqr(series, k=k)
    elif method == "zscore":
        t = threshold if threshold is not None else 3.0
        return detect_outliers_zscore(series, threshold=t)
    elif method == "iforest":
        contamination = threshold if threshold is not None else 0.05
        return detect_outliers_iforest(
            series,
            contamination=contamination,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


def remove_outliers(
    series: pd.Series,
    method: Literal["iqr", "zscore", "iforest"] = "iqr",
    threshold: Optional[float] = None,
    replace_with: Literal["nan", "median", "interpolate"] = "nan",
    **kwargs,
) -> pd.Series:
    """Remove or replace outliers in time series.

    Args:
        series: Input time series.
        method: Detection method.
        threshold: Detection threshold.
        replace_with: How to handle outliers:
            - 'nan': Replace with NaN
            - 'median': Replace with series median
            - 'interpolate': Replace with interpolated values

    Returns:
        Series with outliers removed/replaced.
    """
    outliers = detect_outliers(series, method=method, threshold=threshold, **kwargs)
    result = series.copy()

    n_outliers = outliers.sum()
    if n_outliers > 0:
        logger.debug(f"Detected {n_outliers} outliers ({n_outliers/len(series)*100:.2f}%)")

        if replace_with == "nan":
            result[outliers] = np.nan
        elif replace_with == "median":
            result[outliers] = series.median()
        elif replace_with == "interpolate":
            result[outliers] = np.nan
            result = result.interpolate(method="linear")
        else:
            raise ValueError(f"Unknown replace_with value: {replace_with}")

    return result


def detect_outliers_rolling(
    series: pd.Series,
    window: int = 30,
    threshold: float = 3.0,
) -> pd.Series:
    """Detect outliers using rolling window Z-score.

    Useful for detecting local anomalies in non-stationary data.

    Args:
        series: Input time series.
        window: Rolling window size.
        threshold: Z-score threshold.

    Returns:
        Boolean series indicating outliers.
    """
    rolling_mean = series.rolling(window=window, center=True, min_periods=1).mean()
    rolling_std = series.rolling(window=window, center=True, min_periods=1).std()

    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)

    z_scores = np.abs((series - rolling_mean) / rolling_std)
    return z_scores > threshold
