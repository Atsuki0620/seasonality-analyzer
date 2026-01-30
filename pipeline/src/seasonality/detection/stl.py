"""STL（Loessを使用した季節-トレンド分解）検出器。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from loguru import logger

from seasonality.detection.base import BaseDetector, DetectionResult, PeriodResult


class STLDetector(BaseDetector):
    """STL分解を使用した季節性検出器。

    STLは時系列データをトレンド、季節、残差の各成分に分解します。
    季節性の強度は、季節成分の分散と残差成分の分散の比率で測定されます。
    """

    method_name = "stl"

    def __init__(
        self,
        seasonal_smoother: int = 7,
        robust: bool = True,
        strength_threshold: float = 0.5,
    ):
        """STL検出器を初期化する。

        Args:
            seasonal_smoother: 季節平滑化パラメータ。
            robust: ロバストフィッティングを使用するかどうか。
            strength_threshold: 有意性のしきい値。
        """
        self.seasonal_smoother = seasonal_smoother
        self.robust = robust
        self.strength_threshold = strength_threshold

    def detect(
        self,
        series: pd.Series,
        periods: List[int],
    ) -> DetectionResult:
        """STL分解を使用して季節性を検出する。

        Args:
            series: 入力時系列データ（前処理済み、NaNなし）。
            periods: 確認する対象周期のリスト。

        Returns:
            各周期のSTLスコアを含むDetectionResult。
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
        """特定の周期の季節性を検出する。"""
        # 最小データ長を確認
        if len(series) < 2 * period:
            logger.debug(f"STL: 周期{period}にはデータが短すぎます（{len(series)} < {2 * period}）")
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

            # 季節性の強度（F_season）を計算
            var_seasonal = np.var(result.seasonal)
            var_residual = np.var(result.resid)

            if var_residual > 0:
                f_season = var_seasonal / var_residual
                # 0-1の範囲に正規化
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
            logger.warning(f"周期{period}のSTLが失敗しました: {e}")
            return PeriodResult(
                period=period,
                score=0.0,
                significant=False,
                details={"error": str(e)},
            )

    def _calc_trend_strength(self, stl_result) -> float:
        """STL結果からトレンドの強度を計算する。"""
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
        """STL分解を実行し、成分を返す。

        Args:
            series: 入力時系列データ。
            period: 季節周期。

        Returns:
            'trend'、'seasonal'、'residual'成分を含む辞書。
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
            logger.warning(f"STL分解が失敗しました: {e}")
            return None
