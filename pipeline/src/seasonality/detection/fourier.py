"""フーリエ回帰検出器。"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from loguru import logger

from seasonality.detection.base import BaseDetector, DetectionResult, PeriodResult


class FourierDetector(BaseDetector):
    """フーリエ回帰を使用した季節性検出器。

    対象周期のサイン/コサイン項を適合し、
    R²スコアの改善（デルタR²）を測定します。
    """

    method_name = "fourier"

    def __init__(
        self,
        delta_r2_threshold: float = 0.1,
    ):
        """フーリエ検出器を初期化する。

        Args:
            delta_r2_threshold: 有意性の最小デルタR²。
        """
        self.delta_r2_threshold = delta_r2_threshold

    def detect(
        self,
        series: pd.Series,
        periods: List[int],
    ) -> DetectionResult:
        """フーリエ回帰を使用して季節性を検出する。

        Args:
            series: 入力時系列データ（前処理済み、NaNなし）。
            periods: 確認する対象周期のリスト。

        Returns:
            各周期のデルタR²を含むDetectionResult。
        """
        clean_series = self._validate_series(series)

        # タイムスタンプを日数に変換
        t = (clean_series.index - clean_series.index[0]).days.values.astype(float)
        y = clean_series.values

        # ベースラインモデル（定数のみ）
        X_base = np.ones((len(t), 1))
        model_base = LinearRegression().fit(X_base, y)
        r2_base = r2_score(y, model_base.predict(X_base))

        period_results: Dict[int, PeriodResult] = {}

        # 各周期を個別に評価
        for period in periods:
            result = self._detect_period(t, y, period, r2_base)
            period_results[period] = result

        # すべての周期を含む完全なモデルも計算
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
        """フーリエ回帰で単一周期を評価する。"""
        # フーリエ特徴量を作成
        sin_term = np.sin(2 * np.pi * t / period)
        cos_term = np.cos(2 * np.pi * t / period)

        X = np.column_stack([np.ones(len(t)), sin_term, cos_term])

        try:
            model = LinearRegression().fit(X, y)
            r2 = r2_score(y, model.predict(X))
            delta_r2 = r2 - r2_base

            significant = delta_r2 > self.delta_r2_threshold

            # デルタR²を0-1の範囲に正規化（1で上限）
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
            logger.warning(f"周期{period}のフーリエ回帰が失敗しました: {e}")
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
        """すべての周期を組み合わせたモデルを計算する。"""
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
            logger.warning(f"完全なフーリエモデルが失敗しました: {e}")
            return {"r2": r2_base, "delta_r2": 0.0}

    def fit_and_predict(
        self,
        series: pd.Series,
        periods: List[int],
    ) -> Optional[pd.Series]:
        """フーリエモデルを適合し、予測を返す。

        Args:
            series: 入力時系列データ。
            periods: モデルに含める周期。

        Returns:
            予測値を含むSeries。
        """
        clean_series = self._validate_series(series)

        t = (clean_series.index - clean_series.index[0]).days.values.astype(float)
        y = clean_series.values

        # 特徴量を構築
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
            logger.warning(f"フーリエ予測が失敗しました: {e}")
            return None

    def get_delta_r2(
        self,
        series: pd.Series,
        periods: List[int],
    ) -> Dict[str, float]:
        """分析用のデルタR²値を取得する。

        Args:
            series: 入力時系列データ。
            periods: 対象周期。

        Returns:
            周期（および'full'）からデルタR²へのマッピング辞書。
        """
        result = self.detect(series, periods)

        delta_r2 = {}
        for period, presult in result.periods.items():
            delta_r2[f"period_{period}"] = presult.details.get("delta_r2", 0.0)

        if result.raw_data:
            delta_r2["full"] = result.raw_data.get("full_model_delta_r2", 0.0)

        return delta_r2
