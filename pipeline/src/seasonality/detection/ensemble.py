"""Ensemble detector combining multiple detection methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from seasonality.detection.base import BaseDetector, DetectionResult, PeriodResult
from seasonality.detection.stl import STLDetector
from seasonality.detection.acf import ACFDetector
from seasonality.detection.periodogram import LombScargleDetector
from seasonality.detection.fourier import FourierDetector
from seasonality.config import DetectionConfig


@dataclass
class EnsembleResult:
    """Result from ensemble detection."""

    sensor: str
    results: Dict[str, DetectionResult]
    combined_scores: Dict[int, float]
    best_period: Optional[int]
    max_combined_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sensor": self.sensor,
            "best_period": self.best_period,
            "max_combined_score": self.max_combined_score,
            "combined_scores": self.combined_scores,
            "method_results": {
                method: result.to_dict()
                for method, result in self.results.items()
            },
        }


class EnsembleDetector:
    """Ensemble detector that combines multiple detection methods.

    Combines results from STL, ACF, Lomb-Scargle, and Fourier detectors
    to provide robust seasonality detection.
    """

    def __init__(
        self,
        config: Optional[DetectionConfig] = None,
        methods: Optional[List[str]] = None,
    ):
        """Initialize ensemble detector.

        Args:
            config: Detection configuration.
            methods: List of methods to use. If None, uses config or defaults.
        """
        self.config = config or DetectionConfig()

        if methods is not None:
            self.methods = methods
        else:
            self.methods = self.config.methods

        # Initialize detectors
        self.detectors: Dict[str, BaseDetector] = {}

        if "stl" in self.methods:
            self.detectors["stl"] = STLDetector(
                seasonal_smoother=self.config.stl_seasonal,
                robust=self.config.stl_robust,
            )

        if "acf" in self.methods:
            self.detectors["acf"] = ACFDetector(
                max_lag=self.config.acf_max_lag,
                tolerance=self.config.acf_tolerance,
            )

        if "periodogram" in self.methods:
            self.detectors["periodogram"] = LombScargleDetector(
                min_period=self.config.periodogram_min_period,
                max_period=self.config.periodogram_max_period,
                fap_threshold=self.config.fap_threshold,
            )

        if "fourier" in self.methods:
            self.detectors["fourier"] = FourierDetector()

        logger.debug(f"Initialized EnsembleDetector with methods: {list(self.detectors.keys())}")

    def detect(
        self,
        series: pd.Series,
        periods: Optional[List[int]] = None,
        sensor_name: str = "unknown",
    ) -> EnsembleResult:
        """Run all detection methods on a time series.

        Args:
            series: Input time series (should be preprocessed).
            periods: Target periods. If None, uses config.
            sensor_name: Name of sensor for logging/results.

        Returns:
            EnsembleResult with combined results.
        """
        if periods is None:
            periods = self.config.periods

        results: Dict[str, DetectionResult] = {}

        for method_name, detector in self.detectors.items():
            try:
                result = detector.detect(series, periods)
                results[method_name] = result
                logger.debug(f"{sensor_name}/{method_name}: max_score={result.max_score:.4f}")
            except Exception as e:
                logger.warning(f"{sensor_name}/{method_name} failed: {e}")
                results[method_name] = DetectionResult(
                    method=method_name,
                    periods={},
                    max_score=0.0,
                    best_period=None,
                )

        # Combine scores
        combined_scores = self._combine_scores(results, periods)

        # Find best period
        if combined_scores:
            best_period = max(combined_scores.keys(), key=lambda p: combined_scores[p])
            max_score = combined_scores[best_period]
        else:
            best_period = None
            max_score = 0.0

        return EnsembleResult(
            sensor=sensor_name,
            results=results,
            combined_scores=combined_scores,
            best_period=best_period,
            max_combined_score=max_score,
        )

    def _combine_scores(
        self,
        results: Dict[str, DetectionResult],
        periods: List[int],
    ) -> Dict[int, float]:
        """Combine scores from multiple methods.

        Uses simple averaging for now; can be extended for weighted combination.
        """
        combined = {}

        for period in periods:
            scores = []
            for method, result in results.items():
                score = result.get_score(period)
                scores.append(score)

            if scores:
                combined[period] = sum(scores) / len(scores)
            else:
                combined[period] = 0.0

        return combined

    def detect_batch(
        self,
        df: pd.DataFrame,
        sensor_cols: List[str],
        periods: Optional[List[int]] = None,
    ) -> List[EnsembleResult]:
        """Run detection on multiple sensors.

        Args:
            df: DataFrame with sensor columns.
            sensor_cols: List of sensor column names.
            periods: Target periods.

        Returns:
            List of EnsembleResults.
        """
        from tqdm import tqdm

        results = []

        for sensor in tqdm(sensor_cols, desc="Detecting seasonality"):
            if sensor not in df.columns:
                logger.warning(f"Sensor {sensor} not found in DataFrame")
                continue

            series = df[sensor].dropna()

            if len(series) < 100:
                logger.warning(f"Skipping {sensor}: insufficient data ({len(series)} points)")
                continue

            try:
                result = self.detect(series, periods=periods, sensor_name=sensor)
                results.append(result)
            except Exception as e:
                logger.error(f"Detection failed for {sensor}: {e}")

        return results


def create_detector(config: Optional[DetectionConfig] = None) -> EnsembleDetector:
    """Factory function to create an ensemble detector.

    Args:
        config: Detection configuration.

    Returns:
        Configured EnsembleDetector.
    """
    return EnsembleDetector(config=config)
