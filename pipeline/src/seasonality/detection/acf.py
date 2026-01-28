"""ACF (Autocorrelation Function) detector."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
from loguru import logger

from seasonality.detection.base import BaseDetector, DetectionResult, PeriodResult


class ACFDetector(BaseDetector):
    """Seasonality detector using Autocorrelation Function.

    ACF measures the correlation between a time series and its lagged values.
    Peaks at specific lags indicate periodicity.
    """

    method_name = "acf"

    def __init__(
        self,
        max_lag: int = 400,
        tolerance: int = 3,
        peak_threshold: float = 0.3,
        use_fft: bool = True,
    ):
        """Initialize ACF detector.

        Args:
            max_lag: Maximum lag to compute ACF.
            tolerance: Tolerance for peak detection around target periods.
            peak_threshold: Threshold for significance (above 95% CI).
            use_fft: Whether to use FFT for faster computation.
        """
        self.max_lag = max_lag
        self.tolerance = tolerance
        self.peak_threshold = peak_threshold
        self.use_fft = use_fft

    def detect(
        self,
        series: pd.Series,
        periods: List[int],
    ) -> DetectionResult:
        """Detect seasonality using ACF analysis.

        Args:
            series: Input time series (should be preprocessed, no NaN).
            periods: List of target periods to check.

        Returns:
            DetectionResult with ACF peak values for each period.
        """
        clean_series = self._validate_series(series)
        n = len(clean_series)

        # Adjust max_lag based on data length
        max_lag = min(self.max_lag, n // 2 - 1)
        if max_lag < 10:
            logger.warning(f"ACF: max_lag too small ({max_lag}), results may be unreliable")
            return self._create_result({})

        # Compute ACF
        try:
            acf_values = acf(clean_series.values, nlags=max_lag, fft=self.use_fft)
        except Exception as e:
            logger.warning(f"ACF computation failed: {e}")
            return self._create_result({})

        # 95% confidence interval
        conf_int = 1.96 / np.sqrt(n)

        period_results: Dict[int, PeriodResult] = {}

        for period in periods:
            result = self._detect_period(
                acf_values, period, max_lag, conf_int
            )
            period_results[period] = result

        return self._create_result(
            period_results,
            raw_data={
                "acf_values": acf_values.tolist(),
                "conf_int": conf_int,
                "max_lag": max_lag,
            },
        )

    def _detect_period(
        self,
        acf_values: np.ndarray,
        period: int,
        max_lag: int,
        conf_int: float,
    ) -> PeriodResult:
        """Detect ACF peak for a specific period."""
        if period > max_lag:
            return PeriodResult(
                period=period,
                score=0.0,
                significant=False,
                details={"error": "period_exceeds_max_lag"},
            )

        # Search range around target period
        start = max(1, period - self.tolerance)
        end = min(max_lag, period + self.tolerance)

        # Find peak in range
        acf_range = acf_values[start:end + 1]
        max_idx = np.argmax(np.abs(acf_range))
        peak_value = acf_range[max_idx]
        actual_lag = start + max_idx

        # Check significance
        significant = abs(peak_value) > conf_int and abs(peak_value) > self.peak_threshold

        return PeriodResult(
            period=period,
            score=float(abs(peak_value)),
            significant=significant,
            details={
                "peak_value": float(peak_value),
                "actual_lag": int(actual_lag),
                "conf_int": float(conf_int),
                "above_ci": abs(peak_value) > conf_int,
            },
        )

    def compute_full_acf(
        self,
        series: pd.Series,
        max_lag: Optional[int] = None,
    ) -> Dict[str, any]:
        """Compute full ACF for visualization.

        Args:
            series: Input time series.
            max_lag: Maximum lag (None uses default).

        Returns:
            Dictionary with ACF values and confidence interval.
        """
        clean_series = self._validate_series(series)
        n = len(clean_series)

        if max_lag is None:
            max_lag = min(self.max_lag, n // 2 - 1)
        else:
            max_lag = min(max_lag, n // 2 - 1)

        acf_values = acf(clean_series.values, nlags=max_lag, fft=self.use_fft)
        conf_int = 1.96 / np.sqrt(n)

        return {
            "lags": list(range(max_lag + 1)),
            "acf": acf_values.tolist(),
            "conf_int_upper": conf_int,
            "conf_int_lower": -conf_int,
        }
