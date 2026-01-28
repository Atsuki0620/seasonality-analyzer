"""Lomb-Scargle periodogram detector."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from seasonality.detection.base import BaseDetector, DetectionResult, PeriodResult

try:
    from astropy.timeseries import LombScargle
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False
    logger.warning("astropy not available. LombScargleDetector will be limited.")


class LombScargleDetector(BaseDetector):
    """Seasonality detector using Lomb-Scargle periodogram.

    Lomb-Scargle is suitable for unevenly sampled data and provides
    False Alarm Probability (FAP) for significance testing.
    """

    method_name = "periodogram"

    def __init__(
        self,
        min_period: int = 2,
        max_period: int = 500,
        fap_threshold: float = 0.05,
        samples_per_peak: int = 10,
    ):
        """Initialize Lomb-Scargle detector.

        Args:
            min_period: Minimum period to consider (days).
            max_period: Maximum period to consider (days).
            fap_threshold: False Alarm Probability threshold.
            samples_per_peak: Frequency resolution parameter.
        """
        if not ASTROPY_AVAILABLE:
            logger.warning("LombScargleDetector initialized but astropy not available")

        self.min_period = min_period
        self.max_period = max_period
        self.fap_threshold = fap_threshold
        self.samples_per_peak = samples_per_peak

    def detect(
        self,
        series: pd.Series,
        periods: List[int],
    ) -> DetectionResult:
        """Detect seasonality using Lomb-Scargle periodogram.

        Args:
            series: Input time series (should be preprocessed).
            periods: List of target periods to check.

        Returns:
            DetectionResult with power and FAP for each period.
        """
        if not ASTROPY_AVAILABLE:
            logger.error("astropy required for LombScargleDetector")
            return self._create_result({})

        clean_series = self._validate_series(series)

        # Convert timestamps to days from start
        t = (clean_series.index - clean_series.index[0]).days.astype(float)
        y = clean_series.values

        # Remove mean
        y = y - np.mean(y)

        # Create Lomb-Scargle object
        min_freq = 1 / self.max_period
        max_freq = 1 / self.min_period

        try:
            ls = LombScargle(t, y)
            frequency, power = ls.autopower(
                minimum_frequency=min_freq,
                maximum_frequency=max_freq,
                samples_per_peak=self.samples_per_peak,
            )
            computed_periods = 1 / frequency
        except Exception as e:
            logger.warning(f"Lomb-Scargle computation failed: {e}")
            return self._create_result({})

        period_results: Dict[int, PeriodResult] = {}

        for target_period in periods:
            result = self._detect_period(
                ls, computed_periods, power, target_period
            )
            period_results[target_period] = result

        return self._create_result(
            period_results,
            raw_data={
                "frequencies": frequency.tolist(),
                "powers": power.tolist(),
                "periods": computed_periods.tolist(),
            },
        )

    def _detect_period(
        self,
        ls: "LombScargle",
        periods: np.ndarray,
        powers: np.ndarray,
        target_period: int,
    ) -> PeriodResult:
        """Detect power and significance for a specific period."""
        # Search in ±20% range of target period
        mask = (periods > target_period * 0.8) & (periods < target_period * 1.2)

        if mask.sum() == 0:
            return PeriodResult(
                period=target_period,
                score=0.0,
                significant=False,
                details={"error": "period_out_of_range"},
            )

        # Find peak in range
        range_powers = powers[mask]
        range_periods = periods[mask]
        max_idx = np.argmax(range_powers)

        peak_power = float(range_powers[max_idx])
        actual_period = float(range_periods[max_idx])

        # Calculate False Alarm Probability
        try:
            fap = float(ls.false_alarm_probability(peak_power))
        except Exception:
            fap = 1.0

        significant = fap < self.fap_threshold

        # Normalize power to 0-1 range (rough approximation)
        score = min(peak_power, 1.0)

        return PeriodResult(
            period=target_period,
            score=score,
            significant=significant,
            details={
                "power": peak_power,
                "actual_period": actual_period,
                "fap": fap,
                "fap_threshold": self.fap_threshold,
            },
        )

    def compute_full_periodogram(
        self,
        series: pd.Series,
    ) -> Optional[Dict[str, any]]:
        """Compute full periodogram for visualization.

        Args:
            series: Input time series.

        Returns:
            Dictionary with frequencies, powers, and periods.
        """
        if not ASTROPY_AVAILABLE:
            return None

        clean_series = self._validate_series(series)

        t = (clean_series.index - clean_series.index[0]).days.astype(float)
        y = clean_series.values - np.mean(clean_series.values)

        min_freq = 1 / self.max_period
        max_freq = 1 / self.min_period

        try:
            ls = LombScargle(t, y)
            frequency, power = ls.autopower(
                minimum_frequency=min_freq,
                maximum_frequency=max_freq,
                samples_per_peak=self.samples_per_peak,
            )

            return {
                "frequencies": frequency.tolist(),
                "powers": power.tolist(),
                "periods": (1 / frequency).tolist(),
            }
        except Exception as e:
            logger.warning(f"Periodogram computation failed: {e}")
            return None
