"""ACF（自己相関関数）検出器。"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
from loguru import logger

from seasonality.detection.base import BaseDetector, DetectionResult, PeriodResult


class ACFDetector(BaseDetector):
    """自己相関関数を使用した季節性検出器。

    ACFは時系列データとそのラグ値との相関を測定します。
    特定のラグでのピークは周期性を示します。
    """

    method_name = "acf"

    def __init__(
        self,
        max_lag: int = 400,
        tolerance: int = 3,
        peak_threshold: float = 0.3,
        use_fft: bool = True,
    ):
        """ACF検出器を初期化する。

        Args:
            max_lag: ACFを計算する最大ラグ。
            tolerance: 対象周期周辺のピーク検出の許容範囲。
            peak_threshold: 有意性のしきい値（95%信頼区間以上）。
            use_fft: より高速な計算にFFTを使用するかどうか。
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
        """ACF分析を使用して季節性を検出する。

        Args:
            series: 入力時系列データ（前処理済み、NaNなし）。
            periods: 確認する対象周期のリスト。

        Returns:
            各周期のACFピーク値を含むDetectionResult。
        """
        clean_series = self._validate_series(series)
        n = len(clean_series)

        # データ長に基づいてmax_lagを調整
        max_lag = min(self.max_lag, n // 2 - 1)
        if max_lag < 10:
            logger.warning(f"ACF: max_lagが小さすぎます（{max_lag}）、結果が信頼できない可能性があります")
            return self._create_result({})

        # ACFを計算
        try:
            acf_values = acf(clean_series.values, nlags=max_lag, fft=self.use_fft)
        except Exception as e:
            logger.warning(f"ACF計算が失敗しました: {e}")
            return self._create_result({})

        # 95%信頼区間
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
        """特定の周期のACFピークを検出する。"""
        if period > max_lag:
            return PeriodResult(
                period=period,
                score=0.0,
                significant=False,
                details={"error": "period_exceeds_max_lag"},
            )

        # 対象周期周辺の探索範囲
        start = max(1, period - self.tolerance)
        end = min(max_lag, period + self.tolerance)

        # 範囲内でピークを検索
        acf_range = acf_values[start:end + 1]
        max_idx = np.argmax(np.abs(acf_range))
        peak_value = acf_range[max_idx]
        actual_lag = start + max_idx

        # 有意性を確認
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
        """可視化用の完全なACFを計算する。

        Args:
            series: 入力時系列データ。
            max_lag: 最大ラグ（Noneの場合はデフォルトを使用）。

        Returns:
            ACF値と信頼区間を含む辞書。
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
