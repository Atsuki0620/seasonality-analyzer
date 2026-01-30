"""複数の検出手法を組み合わせたアンサンブル検出器。"""

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
    """アンサンブル検出の結果。"""

    sensor: str
    results: Dict[str, DetectionResult]
    combined_scores: Dict[int, float]
    best_period: Optional[int]
    max_combined_score: float

    def to_dict(self) -> Dict[str, Any]:
        """シリアライゼーション用に辞書に変換する。"""
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
    """複数の検出手法を組み合わせたアンサンブル検出器。

    STL、ACF、ロンブ・スカーグル、フーリエ検出器の結果を組み合わせて、
    ロバストな季節性検出を提供します。
    """

    def __init__(
        self,
        config: Optional[DetectionConfig] = None,
        methods: Optional[List[str]] = None,
    ):
        """アンサンブル検出器を初期化する。

        Args:
            config: 検出設定。
            methods: 使用する手法のリスト。Noneの場合は、configまたはデフォルト値を使用。
        """
        self.config = config or DetectionConfig()

        if methods is not None:
            self.methods = methods
        else:
            self.methods = self.config.methods

        # 検出器を初期化
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

        logger.debug(f"アンサンブル検出器を初期化しました。使用手法: {list(self.detectors.keys())}")

    def detect(
        self,
        series: pd.Series,
        periods: Optional[List[int]] = None,
        sensor_name: str = "unknown",
    ) -> EnsembleResult:
        """時系列データに対してすべての検出手法を実行する。

        Args:
            series: 入力時系列データ（前処理済み）。
            periods: 対象周期。Noneの場合はconfigを使用。
            sensor_name: ログ/結果用のセンサー名。

        Returns:
            統合結果を含むEnsembleResult。
        """
        if periods is None:
            periods = self.config.periods

        results: Dict[str, DetectionResult] = {}

        for method_name, detector in self.detectors.items():
            try:
                result = detector.detect(series, periods)
                results[method_name] = result
                logger.debug(f"{sensor_name}/{method_name}: 最大スコア={result.max_score:.4f}")
            except Exception as e:
                logger.warning(f"{sensor_name}/{method_name} が失敗しました: {e}")
                results[method_name] = DetectionResult(
                    method=method_name,
                    periods={},
                    max_score=0.0,
                    best_period=None,
                )

        # スコアを統合
        combined_scores = self._combine_scores(results, periods)

        # 最適周期を検索
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
        """複数の手法からのスコアを統合する。

        現在は単純平均を使用しています。重み付き統合に拡張可能です。
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
        """複数のセンサーに対して検出を実行する。

        Args:
            df: センサー列を含むDataFrame。
            sensor_cols: センサー列名のリスト。
            periods: 対象周期。

        Returns:
            EnsembleResultのリスト。
        """
        from tqdm import tqdm

        results = []

        for sensor in tqdm(sensor_cols, desc="季節性を検出中"):
            if sensor not in df.columns:
                logger.warning(f"センサー {sensor} がDataFrameに見つかりません")
                continue

            series = df[sensor].dropna()

            if len(series) < 100:
                logger.warning(f"{sensor}をスキップします: データ不足（{len(series)}ポイント）")
                continue

            try:
                result = self.detect(series, periods=periods, sensor_name=sensor)
                results.append(result)
            except Exception as e:
                logger.error(f"{sensor}の検出が失敗しました: {e}")

        return results


def create_detector(config: Optional[DetectionConfig] = None) -> EnsembleDetector:
    """アンサンブル検出器を作成するファクトリー関数。

    Args:
        config: 検出設定。

    Returns:
        設定されたEnsembleDetector。
    """
    return EnsembleDetector(config=config)
