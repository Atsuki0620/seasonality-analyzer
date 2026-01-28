"""Fourier regression detector."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from loguru import logger

from seasonality.detection.base import BaseDetector, DetectionResult, PeriodResult


class FourierDetector(BaseDetector):
    """Seasonality detector using Fourier regression.

    Fits sine/cosine terms for target periods and measures
    improvement in R² score (delta R²).
    """

    method_name = "fourier"

    def __init__(
        self,
        delta_r2_threshold: float = 0.1,
    ):
        """Initialize Fourier detector.

        Args:
            delta_r2_threshold: Minimum delta R² for significance.
        """
        self.delta_r2_threshold = delta_r2_threshold

    def detect(
        self,
        series: pd.Series,
        periods: List[int],
    ) -> DetectionResult:
        """Detect seasonality using Fourier regression.

        Args:
            series: Input time series (should be preprocessed, no NaN).
            periods: List of target periods to check.

        Returns:
            DetectionResult with delta R² for each period.
        """
        clean_series = self._validate_series(series)

        # Convert timestamps to days
        t = (clean_series.index - clean_series.index[0]).days.values.astype(float)
        y = clean_series.values

        # Baseline model (constant only)
        X_base = np.ones((len(t), 1))
        model_base = LinearRegression().fit(X_base, y)
        r2_base = r2_score(y, model_base.predict(X_base))

        period_results: Dict[int, PeriodResult] = {}

        # Evaluate each period individually
        for period in periods:
            result = self._detect_period(t, y, period, r2_base)
            period_results[period] = result

        # Also compute full model with all periods
        full_result = self._compute_full_model(t, y, periods, r2_base)

        return self._create_result(
            period_results,
            raw_data={
                "baseline_r2": r2_base,
                "full_model_r2": full_result["r2"],
                "full_model_delta_r2": full_result["delta_r2"],
            },
        )

    def _detect_period(
        self,
        t: np.ndarray,
        y: np.ndarray,
        period: int,
        r2_base: float,
    ) -> PeriodResult:
        """Evaluate single period with Fourier regression."""
        # Create Fourier features
        sin_term = np.sin(2 * np.pi * t / period)
        cos_term = np.cos(2 * np.pi * t / period)

        X = np.column_stack([np.ones(len(t)), sin_term, cos_term])

        try:
            model = LinearRegression().fit(X, y)
            r2 = r2_score(y, model.predict(X))
            delta_r2 = r2 - r2_base

            significant = delta_r2 > self.delta_r2_threshold

            # Normalize delta_r2 to 0-1 range (cap at 1)
            score = min(max(delta_r2 * 10, 0), 1)

            return PeriodResult(
                period=period,
                score=score,
                significant=significant,
                details={
                    "r2": float(r2),
                    "delta_r2": float(delta_r2),
                    "baseline_r2": float(r2_base),
                    "sin_coef": float(model.coef_[1]),
                    "cos_coef": float(model.coef_[2]),
                },
            )

        except Exception as e:
            logger.warning(f"Fourier regression failed for period {period}: {e}")
            return PeriodResult(
                period=period,
                score=0.0,
                significant=False,
                details={"error": str(e)},
            )

    def _compute_full_model(
        self,
        t: np.ndarray,
        y: np.ndarray,
        periods: List[int],
        r2_base: float,
    ) -> Dict[str, float]:
        """Compute model with all periods combined."""
        X_full = [np.ones(len(t))]

        for period in periods:
            X_full.append(np.sin(2 * np.pi * t / period))
            X_full.append(np.cos(2 * np.pi * t / period))

        X_full = np.column_stack(X_full)

        try:
            model = LinearRegression().fit(X_full, y)
            r2 = r2_score(y, model.predict(X_full))
            delta_r2 = r2 - r2_base

            return {
                "r2": float(r2),
                "delta_r2": float(delta_r2),
            }
        except Exception as e:
            logger.warning(f"Full Fourier model failed: {e}")
            return {"r2": r2_base, "delta_r2": 0.0}

    def fit_and_predict(
        self,
        series: pd.Series,
        periods: List[int],
    ) -> Optional[pd.Series]:
        """Fit Fourier model and return predictions.

        Args:
            series: Input time series.
            periods: Periods to include in model.

        Returns:
            Predicted values as Series.
        """
        clean_series = self._validate_series(series)

        t = (clean_series.index - clean_series.index[0]).days.values.astype(float)
        y = clean_series.values

        # Build features
        X = [np.ones(len(t))]
        for period in periods:
            X.append(np.sin(2 * np.pi * t / period))
            X.append(np.cos(2 * np.pi * t / period))
        X = np.column_stack(X)

        try:
            model = LinearRegression().fit(X, y)
            predictions = model.predict(X)
            return pd.Series(predictions, index=clean_series.index)
        except Exception as e:
            logger.warning(f"Fourier prediction failed: {e}")
            return None

    def get_delta_r2(
        self,
        series: pd.Series,
        periods: List[int],
    ) -> Dict[str, float]:
        """Get delta R² values for analysis.

        Args:
            series: Input time series.
            periods: Target periods.

        Returns:
            Dictionary mapping period (and 'full') to delta R².
        """
        result = self.detect(series, periods)

        delta_r2 = {}
        for period, presult in result.periods.items():
            delta_r2[f"period_{period}"] = presult.details.get("delta_r2", 0.0)

        if result.raw_data:
            delta_r2["full"] = result.raw_data.get("full_model_delta_r2", 0.0)

        return delta_r2
