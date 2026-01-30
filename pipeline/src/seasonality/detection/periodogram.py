"""ロンブ・スカーグル・ペリオドグラム検出器。"""

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
    logger.warning("astropyが利用できません。LombScargleDetectorの機能が制限されます。")


class LombScargleDetector(BaseDetector):
    """ロンブ・スカーグル・ペリオドグラムを使用した季節性検出器。

    ロンブ・スカーグルは不均等にサンプリングされたデータに適しており、
    有意性検定のための誤警報確率（FAP）を提供します。
    """

    method_name = "periodogram"

    def __init__(
        self,
        min_period: int = 2,
        max_period: int = 500,
        fap_threshold: float = 0.05,
        samples_per_peak: int = 10,
    ):
        """ロンブ・スカーグル検出器を初期化する。

        Args:
            min_period: 考慮する最小周期（日数）。
            max_period: 考慮する最大周期（日数）。
            fap_threshold: 誤警報確率のしきい値。
            samples_per_peak: 周波数分解能パラメータ。
        """
        if not ASTROPY_AVAILABLE:
            logger.warning("LombScargleDetectorが初期化されましたが、astropyが利用できません")

        self.min_period = min_period
        self.max_period = max_period
        self.fap_threshold = fap_threshold
        self.samples_per_peak = samples_per_peak

    def detect(
        self,
        series: pd.Series,
        periods: List[int],
    ) -> DetectionResult:
        """ロンブ・スカーグル・ペリオドグラムを使用して季節性を検出する。

        Args:
            series: 入力時系列データ（前処理済み）。
            periods: 確認する対象周期のリスト。

        Returns:
            各周期のパワーとFAPを含むDetectionResult。
        """
        if not ASTROPY_AVAILABLE:
            logger.error("LombScargleDetectorにはastropyが必要です")
            return self._create_result({})

        clean_series = self._validate_series(series)

        # タイムスタンプを開始からの日数に変換
        t = (clean_series.index - clean_series.index[0]).days.astype(float)
        y = clean_series.values

        # 平均を除去
        y = y - np.mean(y)

        # ロンブ・スカーグルオブジェクトを作成
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
            logger.warning(f"ロンブ・スカーグル計算が失敗しました: {e}")
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
        """特定の周期のパワーと有意性を検出する。"""
        # 対象周期の±20%の範囲で検索
        mask = (periods > target_period * 0.8) & (periods < target_period * 1.2)

        if mask.sum() == 0:
            return PeriodResult(
                period=target_period,
                score=0.0,
                significant=False,
                details={"error": "period_out_of_range"},
            )

        # 範囲内でピークを検索
        range_powers = powers[mask]
        range_periods = periods[mask]
        max_idx = np.argmax(range_powers)

        peak_power = float(range_powers[max_idx])
        actual_period = float(range_periods[max_idx])

        # 誤警報確率を計算
        try:
            fap = float(ls.false_alarm_probability(peak_power))
        except Exception:
            fap = 1.0

        significant = fap < self.fap_threshold

        # パワーを0-1の範囲に正規化（おおよその近似）
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
        """可視化用の完全なペリオドグラムを計算する。

        Args:
            series: 入力時系列データ。

        Returns:
            周波数、パワー、周期を含む辞書。
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
            logger.warning(f"ペリオドグラム計算が失敗しました: {e}")
            return None
