"""Synthetic data generation for testing."""

from __future__ import annotations

from typing import List, Literal, Optional

import numpy as np
import pandas as pd


def generate_seasonal_data(
    n_days: int = 1460,  # 4 years
    periods: List[int] = [7, 365],
    trend: Literal["none", "linear", "quadratic"] = "linear",
    noise_level: float = 0.1,
    outlier_rate: float = 0.01,
    missing_rate: float = 0.005,
    changepoints: Optional[List[int]] = None,
    base_value: float = 100.0,
    seasonal_amplitudes: Optional[List[float]] = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic seasonal time series data.

    Args:
        n_days: Number of days to generate.
        periods: List of seasonal periods (in days).
        trend: Type of trend ('none', 'linear', 'quadratic').
        noise_level: Standard deviation of noise (relative to base_value).
        outlier_rate: Proportion of outliers to inject.
        missing_rate: Proportion of missing values.
        changepoints: List of day indices where level shifts occur.
        base_value: Base value for the series.
        seasonal_amplitudes: Amplitude for each period (default: 10% of base_value).
        random_seed: Random seed for reproducibility.

    Returns:
        DataFrame with timestamp index and 'value' column.
    """
    np.random.seed(random_seed)

    # Generate date range
    dates = pd.date_range(start="2024-01-01", periods=n_days, freq="D")
    t = np.arange(n_days)

    # Initialize with base value
    values = np.full(n_days, base_value, dtype=float)

    # Add trend
    if trend == "linear":
        values += t * 0.01 * base_value / n_days * 100  # Gentle upward trend
    elif trend == "quadratic":
        values += (t ** 2) * 0.001 * base_value / (n_days ** 2) * 100

    # Add seasonal components
    if seasonal_amplitudes is None:
        seasonal_amplitudes = [base_value * 0.1] * len(periods)

    for period, amplitude in zip(periods, seasonal_amplitudes):
        values += amplitude * np.sin(2 * np.pi * t / period)

    # Add changepoints (level shifts)
    if changepoints:
        for cp in changepoints:
            if cp < n_days:
                shift = np.random.uniform(-0.2, 0.2) * base_value
                values[cp:] += shift

    # Add noise
    noise = np.random.randn(n_days) * noise_level * base_value
    values += noise

    # Add outliers
    n_outliers = int(n_days * outlier_rate)
    outlier_idx = np.random.choice(n_days, n_outliers, replace=False)
    outlier_magnitude = np.random.choice([-1, 1], n_outliers) * np.random.uniform(2, 5, n_outliers) * base_value * 0.2
    values[outlier_idx] += outlier_magnitude

    # Create DataFrame
    df = pd.DataFrame({
        "timestamp": dates,
        "value": values,
    }).set_index("timestamp")

    # Add missing values
    n_missing = int(n_days * missing_rate)
    missing_idx = np.random.choice(n_days, n_missing, replace=False)
    df.iloc[missing_idx, 0] = np.nan

    return df


def generate_multi_sensor_data(
    n_days: int = 730,  # 2 years
    n_sensors: int = 10,
    n_with_seasonality: int = 5,
    periods: List[int] = [7, 30, 365],
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate multi-sensor synthetic data.

    Args:
        n_days: Number of days.
        n_sensors: Total number of sensors.
        n_with_seasonality: Number of sensors with seasonality.
        periods: Seasonal periods.
        random_seed: Random seed.

    Returns:
        DataFrame with timestamp index and sensor columns.
    """
    np.random.seed(random_seed)

    dates = pd.date_range(start="2024-01-01", periods=n_days, freq="D")
    t = np.arange(n_days)

    df = pd.DataFrame(index=dates)
    df.index.name = "timestamp"

    for i in range(n_sensors):
        base = np.random.uniform(50, 150)
        noise = np.random.randn(n_days) * base * 0.1

        if i < n_with_seasonality:
            # Add seasonality
            seasonal = np.zeros(n_days)
            for period in periods:
                amplitude = base * np.random.uniform(0.05, 0.2)
                phase = np.random.uniform(0, 2 * np.pi)
                seasonal += amplitude * np.sin(2 * np.pi * t / period + phase)
            df[f"sensor_{i+1}"] = base + seasonal + noise
        else:
            # No seasonality
            df[f"sensor_{i+1}"] = base + noise

        # Add some missing values
        missing_idx = np.random.choice(n_days, int(n_days * 0.02), replace=False)
        df.iloc[missing_idx, -1] = np.nan

    return df


def generate_csv_file(
    path: str,
    n_days: int = 365,
    n_sensors: int = 3,
    include_time: bool = True,
) -> None:
    """Generate CSV file with synthetic data.

    Args:
        path: Output file path.
        n_days: Number of days.
        n_sensors: Number of sensors.
        include_time: Whether to include time column.
    """
    df = generate_multi_sensor_data(n_days=n_days, n_sensors=n_sensors)
    df = df.reset_index()

    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    if include_time:
        df["time"] = df["timestamp"].dt.strftime("%H:%M:%S")
        cols = ["date", "time"] + [c for c in df.columns if c.startswith("sensor_")]
    else:
        cols = ["date"] + [c for c in df.columns if c.startswith("sensor_")]

    df[cols].to_csv(path, index=False)
