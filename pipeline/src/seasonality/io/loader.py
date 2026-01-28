"""Data loading utilities for Seasonality Analyzer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from loguru import logger

from seasonality.config import DataConfig


@dataclass
class DataHealthReport:
    """Health report for loaded data."""

    total_records: int
    date_range: Tuple[str, str]
    sensor_count: int
    sensor_columns: List[str]
    missing_counts: dict[str, int]
    missing_rate: float
    observations_per_day: dict[str, float]  # min, max, mean
    warnings: List[str] = field(default_factory=list)

    def is_healthy(self) -> bool:
        """Check if data passes basic health checks."""
        return len(self.warnings) == 0 and self.total_records > 0

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=== Data Health Report ===",
            f"Total records: {self.total_records:,}",
            f"Date range: {self.date_range[0]} to {self.date_range[1]}",
            f"Sensor count: {self.sensor_count}",
            f"Overall missing rate: {self.missing_rate:.2%}",
            f"Observations per day: min={self.observations_per_day['min']:.1f}, "
            f"max={self.observations_per_day['max']:.1f}, mean={self.observations_per_day['mean']:.1f}",
        ]
        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)


def auto_detect_sensor_columns(df: pd.DataFrame, prefix: str = "sensor_") -> List[str]:
    """Auto-detect sensor columns based on prefix.

    Args:
        df: Input DataFrame.
        prefix: Column name prefix to match.

    Returns:
        List of column names that match the prefix.
    """
    return [col for col in df.columns if col.startswith(prefix)]


def load_sensor_data(
    path: Path | str,
    config: Optional[DataConfig] = None,
) -> Tuple[pd.DataFrame, DataHealthReport]:
    """Load sensor data from CSV file.

    Args:
        path: Path to CSV file.
        config: Data configuration. If None, uses defaults.

    Returns:
        Tuple of (DataFrame with timestamp index, DataHealthReport).

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    config = config or DataConfig()

    logger.info(f"Loading data from: {path}")

    # Load CSV
    df = pd.read_csv(path)
    logger.debug(f"Loaded {len(df)} records with columns: {list(df.columns)}")

    # Validate required columns
    if config.date_column not in df.columns:
        raise ValueError(f"Date column '{config.date_column}' not found in data")

    # Generate timestamp
    if config.time_column and config.time_column in df.columns:
        df["timestamp"] = pd.to_datetime(
            df[config.date_column] + " " + df[config.time_column]
        )
    else:
        df["timestamp"] = pd.to_datetime(df[config.date_column])

    # Set timestamp as index
    df = df.set_index("timestamp").sort_index()

    # Drop original date/time columns
    cols_to_drop = [col for col in [config.date_column, config.time_column] if col in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Detect sensor columns
    if config.sensor_columns:
        sensor_cols = config.sensor_columns
        missing_cols = [col for col in sensor_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Sensor columns not found: {missing_cols}")
    else:
        sensor_cols = auto_detect_sensor_columns(df, config.sensor_prefix)
        if not sensor_cols:
            raise ValueError(
                f"No sensor columns found with prefix '{config.sensor_prefix}'. "
                f"Available columns: {list(df.columns)}"
            )

    logger.info(f"Detected {len(sensor_cols)} sensor columns")

    # Generate health report
    health_report = _generate_health_report(df, sensor_cols)
    logger.info(health_report.summary())

    return df, health_report


def _generate_health_report(df: pd.DataFrame, sensor_cols: List[str]) -> DataHealthReport:
    """Generate health report for loaded data."""
    warnings = []

    # Basic stats
    total_records = len(df)
    date_range = (str(df.index.min()), str(df.index.max()))
    sensor_count = len(sensor_cols)

    # Missing values
    missing_counts = {col: int(df[col].isna().sum()) for col in sensor_cols}
    total_cells = total_records * sensor_count
    total_missing = sum(missing_counts.values())
    missing_rate = total_missing / total_cells if total_cells > 0 else 0.0

    # Observations per day
    daily_counts = df.groupby(df.index.date).size()
    obs_per_day = {
        "min": float(daily_counts.min()) if len(daily_counts) > 0 else 0.0,
        "max": float(daily_counts.max()) if len(daily_counts) > 0 else 0.0,
        "mean": float(daily_counts.mean()) if len(daily_counts) > 0 else 0.0,
    }

    # Generate warnings
    if total_records == 0:
        warnings.append("No records in data")

    if missing_rate > 0.1:
        warnings.append(f"High missing rate: {missing_rate:.1%}")

    # Check for sensors with all missing values
    all_missing_sensors = [col for col, count in missing_counts.items() if count == total_records]
    if all_missing_sensors:
        warnings.append(f"Sensors with all missing values: {all_missing_sensors}")

    # Check date range
    if total_records > 0:
        date_span = (df.index.max() - df.index.min()).days
        if date_span < 30:
            warnings.append(f"Short date range: {date_span} days (recommended: >30 days)")
        if date_span < 365 and any(p >= 365 for p in [7, 30, 365]):  # Common periods
            warnings.append(f"Date range ({date_span} days) may be insufficient for annual seasonality detection")

    return DataHealthReport(
        total_records=total_records,
        date_range=date_range,
        sensor_count=sensor_count,
        sensor_columns=sensor_cols,
        missing_counts=missing_counts,
        missing_rate=missing_rate,
        observations_per_day=obs_per_day,
        warnings=warnings,
    )


def validate_data_for_analysis(
    df: pd.DataFrame,
    sensor_cols: List[str],
    min_records: int = 100,
) -> Tuple[List[str], List[str]]:
    """Validate which sensors have sufficient data for analysis.

    Args:
        df: Input DataFrame.
        sensor_cols: List of sensor columns.
        min_records: Minimum required non-null records.

    Returns:
        Tuple of (valid_sensors, skipped_sensors).
    """
    valid_sensors = []
    skipped_sensors = []

    for col in sensor_cols:
        non_null_count = df[col].notna().sum()
        if non_null_count >= min_records:
            valid_sensors.append(col)
        else:
            skipped_sensors.append(col)
            logger.warning(f"Skipping {col}: insufficient data ({non_null_count} < {min_records})")

    return valid_sensors, skipped_sensors
