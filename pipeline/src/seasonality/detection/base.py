"""Base classes and protocols for seasonality detection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

import pandas as pd


@dataclass
class PeriodResult:
    """Result for a specific period detection."""

    period: int
    score: float
    significant: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionResult:
    """Result of seasonality detection."""

    method: str
    periods: Dict[int, PeriodResult]
    max_score: float
    best_period: Optional[int]
    raw_data: Optional[Dict[str, Any]] = None

    def get_score(self, period: int) -> float:
        """Get score for specific period."""
        if period in self.periods:
            return self.periods[period].score
        return 0.0

    def is_significant(self, period: int) -> bool:
        """Check if specific period is significant."""
        if period in self.periods:
            return self.periods[period].significant
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method,
            "max_score": self.max_score,
            "best_period": self.best_period,
            "periods": {
                p: {
                    "score": r.score,
                    "significant": r.significant,
                    "details": r.details,
                }
                for p, r in self.periods.items()
            },
        }


class SeasonalityDetector(Protocol):
    """Protocol for seasonality detection methods."""

    def detect(
        self,
        series: pd.Series,
        periods: List[int],
    ) -> DetectionResult:
        """Detect seasonality in time series.

        Args:
            series: Input time series (should be preprocessed).
            periods: List of target periods to check.

        Returns:
            DetectionResult with scores for each period.
        """
        ...


class BaseDetector(ABC):
    """Abstract base class for seasonality detectors."""

    method_name: str = "base"

    @abstractmethod
    def detect(
        self,
        series: pd.Series,
        periods: List[int],
    ) -> DetectionResult:
        """Detect seasonality in time series."""
        pass

    def _validate_series(self, series: pd.Series) -> pd.Series:
        """Validate and clean input series."""
        if series.empty:
            raise ValueError("Input series is empty")

        # Drop NaN values
        clean_series = series.dropna()
        if len(clean_series) == 0:
            raise ValueError("Series contains only NaN values")

        return clean_series

    def _create_result(
        self,
        periods: Dict[int, PeriodResult],
        raw_data: Optional[Dict[str, Any]] = None,
    ) -> DetectionResult:
        """Create DetectionResult from period results."""
        if not periods:
            return DetectionResult(
                method=self.method_name,
                periods={},
                max_score=0.0,
                best_period=None,
                raw_data=raw_data,
            )

        max_score = max(p.score for p in periods.values())
        best_period = max(periods.keys(), key=lambda p: periods[p].score)

        return DetectionResult(
            method=self.method_name,
            periods=periods,
            max_score=max_score,
            best_period=best_period,
            raw_data=raw_data,
        )
