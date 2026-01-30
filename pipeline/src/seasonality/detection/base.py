"""季節性検出のための基底クラスとプロトコル。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

import pandas as pd


@dataclass
class PeriodResult:
    """特定の周期検出の結果。"""

    period: int
    score: float
    significant: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionResult:
    """季節性検出の結果。"""

    method: str
    periods: Dict[int, PeriodResult]
    max_score: float
    best_period: Optional[int]
    raw_data: Optional[Dict[str, Any]] = None

    def get_score(self, period: int) -> float:
        """特定の周期のスコアを取得する。"""
        if period in self.periods:
            return self.periods[period].score
        return 0.0

    def is_significant(self, period: int) -> bool:
        """特定の周期が有意かどうかを確認する。"""
        if period in self.periods:
            return self.periods[period].significant
        return False

    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換する。"""
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
    """季節性検出手法のプロトコル。"""

    def detect(
        self,
        series: pd.Series,
        periods: List[int],
    ) -> DetectionResult:
        """時系列データの季節性を検出する。

        Args:
            series: 入力時系列データ（前処理済みであること）。
            periods: 確認する対象周期のリスト。

        Returns:
            各周期のスコアを含むDetectionResult。
        """
        ...


class BaseDetector(ABC):
    """季節性検出器の抽象基底クラス。"""

    method_name: str = "base"

    @abstractmethod
    def detect(
        self,
        series: pd.Series,
        periods: List[int],
    ) -> DetectionResult:
        """時系列データの季節性を検出する。"""
        pass

    def _validate_series(self, series: pd.Series) -> pd.Series:
        """入力時系列データの検証とクリーニング。"""
        if series.empty:
            raise ValueError("入力時系列データが空です")

        # NaN値を削除
        clean_series = series.dropna()
        if len(clean_series) == 0:
            raise ValueError("時系列データにNaN値のみが含まれています")

        return clean_series

    def _create_result(
        self,
        periods: Dict[int, PeriodResult],
        raw_data: Optional[Dict[str, Any]] = None,
    ) -> DetectionResult:
        """周期結果からDetectionResultを作成する。"""
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
