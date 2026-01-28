"""STL (Seasonal-Trend decomposition using Loess) detector."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from loguru import logger

from seasonality.detection.base import BaseDetector, DetectionResult, PeriodResult


class STLDetector(BaseDetector):
    """Seasonality detector using STL decomposition.

    STL decomposes a time series into Trend, Seasonal, and Residual components.
    The seasonal strength is measured by the ratio of seasonal variance to residual variance.
    """

    method_name = "stl"

    def __init__(
        self,
        seasonal_smoother: int = 7,
        robust: bool = True,
        strength_threshold: float = 0.5,
    ):
        """Initialize STL detector.

        Args:
            seasonal_smoother: Seasonal smoothing parameter.
            robust: Whether to use robust fitting.
            strength_threshold: Threshold for significance.
        """
        self.seasonal_smoother = seasonal_smoother
        self.robust = robust
        self.strength_threshold = strength_threshold

    def detect(
        self,
        series: pd.Series,
        periods: List[int],
    ) -> DetectionResult:
        """Detect seasonality using STL decomposition.

        Args:
            series: Input time series (should be preprocessed, no NaN).
            periods: List of target periods to check.

        Returns:
            DetectionResult with STL scores for each period.
        """
        clean_series = self._validate_series(series)
        period_results: Dict[int, PeriodResult] = {}

        for period in periods:
            result = self._detect_period(clean_series, period)
            if result is not None:
                period_results[period] = result

        return self._create_result(period_results)

    def _detect_period(
        self,
        series: pd.Series,
        period: int,
    ) -> Optional[PeriodResult]:
        """Detect seasonality for a specific period."""
        # Check minimum data length
        if len(series) < 2 * period:
            logger.debug(f"STL: Data too short for period {period} ({len(series)} < {2 * period})")
            return PeriodResult(
                period=period,
                score=0.0,
                significant=False,
                details={"error": "insufficient_data"},
            )

        try:
            stl = STL(
                series,
                period=period,
                seasonal=self.seasonal_smoother,
                robust=self.robust,
            )
            result = stl.fit()

            # Calculate seasonal strength (F_season)
            var_seasonal = np.var(result.seasonal)
            var_residual = np.var(result.resid)

            if var_residual > 0:
                f_season = var_seasonal / var_residual
                # Normalize to 0-1 range
                score = f_season / (f_season + 1)
            else:
                score = 0.0

            significant = score > self.strength_threshold

            return PeriodResult(
                period=period,
                score=float(score),
                significant=significant,
                details={
                    "f_season": float(f_season) if var_residual > 0 else 0.0,
                    "var_seasonal": float(var_seasonal),
                    "var_residual": float(var_residual),
                    "trend_strength": self._calc_trend_strength(result),
                },
            )

        except Exception as e:
            logger.warning(f"STL failed for period {period}: {e}")
            return PeriodResult(
                period=period,
                score=0.0,
                significant=False,
                details={"error": str(e)},
            )

    def _calc_trend_strength(self, stl_result) -> float:
        """Calculate trend strength from STL result."""
        var_trend = np.var(stl_result.trend)
        var_residual = np.var(stl_result.resid)

        if var_residual > 0:
            f_trend = var_trend / var_residual
            return f_trend / (f_trend + 1)
        return 0.0

    def decompose(
        self,
        series: pd.Series,
        period: int,
    ) -> Optional[Dict[str, pd.Series]]:
        """Perform STL decomposition and return components.

        Args:
            series: Input time series.
            period: Seasonal period.

        Returns:
            Dictionary with 'trend', 'seasonal', 'residual' components.
        """
        clean_series = self._validate_series(series)

        if len(clean_series) < 2 * period:
            return None

        try:
            stl = STL(
                clean_series,
                period=period,
                seasonal=self.seasonal_smoother,
                robust=self.robust,
            )
            result = stl.fit()

            return {
                "trend": result.trend,
                "seasonal": result.seasonal,
                "residual": result.resid,
                "observed": clean_series,
            }

        except Exception as e:
            logger.warning(f"STL decomposition failed: {e}")
            return None
