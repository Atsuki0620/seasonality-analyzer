"""Seasonality Analyzer - CLI tool for detecting seasonality patterns in time series data."""

__version__ = "0.1.0"
__author__ = "Seasonality Analyzer Team"

from seasonality.config import (
    SeasonalityConfig,
    DataConfig,
    PreprocessingConfig,
    DetectionConfig,
    ScoringConfig,
    DebugConfig,
    OutputConfig,
)

__all__ = [
    "__version__",
    "SeasonalityConfig",
    "DataConfig",
    "PreprocessingConfig",
    "DetectionConfig",
    "ScoringConfig",
    "DebugConfig",
    "OutputConfig",
]
