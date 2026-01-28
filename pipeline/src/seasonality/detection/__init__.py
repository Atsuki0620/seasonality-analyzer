"""Detection module for Seasonality Analyzer."""

from seasonality.detection.base import (
    SeasonalityDetector,
    DetectionResult,
    PeriodResult,
)
from seasonality.detection.stl import STLDetector
from seasonality.detection.acf import ACFDetector
from seasonality.detection.periodogram import LombScargleDetector
from seasonality.detection.fourier import FourierDetector
from seasonality.detection.ensemble import EnsembleDetector

__all__ = [
    "SeasonalityDetector",
    "DetectionResult",
    "PeriodResult",
    "STLDetector",
    "ACFDetector",
    "LombScargleDetector",
    "FourierDetector",
    "EnsembleDetector",
]
