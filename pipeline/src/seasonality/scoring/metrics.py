"""季節性分析のスコアリングメトリクス。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from seasonality.detection.ensemble import EnsembleResult
from seasonality.config import ScoringConfig


def normalize_score(
    value: float,
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> float:
    """スコアを0-1の範囲に正規化します。

    Args:
        value: 入力値
        min_val: 期待される最小値
        max_val: 期待される最大値

    Returns:
        0から1の間に正規化された値
    """
    if np.isnan(value):
        return 0.0

    if max_val <= min_val:
        return 0.0

    normalized = (value - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, normalized))


def aggregate_method_scores(
    ensemble_result: EnsembleResult,
    periods: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """アンサンブル結果のスコアをフラットな辞書に集約します。

    Args:
        ensemble_result: アンサンブル検出の結果
        periods: 含める周期のリスト。Noneの場合は全て使用

    Returns:
        集約されたスコアを含む辞書
    """
    scores = {"sensor": ensemble_result.sensor}

    # 指定されていない場合は結果から周期を取得
    if periods is None:
        all_periods = set()
        for result in ensemble_result.results.values():
            all_periods.update(result.periods.keys())
        periods = sorted(all_periods)

    # 各手法と周期のスコアを抽出
    for method, result in ensemble_result.results.items():
        max_score = 0.0

        for period in periods:
            period_score = result.get_score(period)
            key = f"{method}_{period}"
            scores[key] = period_score
            max_score = max(max_score, period_score)

        scores[f"{method}_max"] = max_score

    # 統合スコアを追加
    for period in periods:
        scores[f"combined_{period}"] = ensemble_result.combined_scores.get(period, 0.0)

    scores["best_period"] = ensemble_result.best_period
    scores["max_combined_score"] = ensemble_result.max_combined_score

    return scores


def calculate_composite_score(
    scores: Dict[str, float],
    config: Optional[ScoringConfig] = None,
) -> float:
    """手法スコアから重み付き総合スコアを計算します。

    Args:
        scores: 手法スコアを含む辞書（例: 'stl_max', 'acf_max', 'fourier_full'）
        config: 重みを含むスコアリング設定

    Returns:
        重み付き総合スコア
    """
    if config is None:
        weights = {"stl": 0.3, "acf": 0.3, "fourier": 0.4}
    else:
        weights = config.weights

    stl_score = scores.get("stl_max", 0.0)
    acf_score = scores.get("acf_max", 0.0)
    fourier_score = scores.get("fourier_max", 0.0)

    # フーリエスコアを正規化（delta R²は通常小さい値）
    fourier_normalized = min(fourier_score * 10, 1.0)

    composite = (
        weights.get("stl", 0.3) * stl_score +
        weights.get("acf", 0.3) * acf_score +
        weights.get("fourier", 0.4) * fourier_normalized
    )

    return composite


def calculate_all_scores(
    ensemble_result: EnsembleResult,
    config: Optional[ScoringConfig] = None,
) -> Dict[str, Any]:
    """単一センサーの全スコアを計算します。

    Args:
        ensemble_result: アンサンブル検出の結果
        config: スコアリング設定

    Returns:
        計算された全スコアを含む辞書
    """
    config = config or ScoringConfig()

    # 集約された手法スコアを取得
    scores = aggregate_method_scores(ensemble_result)

    # 総合スコアを計算
    scores["composite_score"] = calculate_composite_score(scores, config)

    return scores


def calculate_period_confidence(
    ensemble_result: EnsembleResult,
    period: int,
) -> Dict[str, Any]:
    """特定周期の信頼度メトリクスを計算します。

    Args:
        ensemble_result: アンサンブル検出結果
        period: 対象周期

    Returns:
        周期固有の信頼度メトリクスを含む辞書
    """
    method_scores = {}
    significant_methods = []

    for method, result in ensemble_result.results.items():
        score = result.get_score(period)
        significant = result.is_significant(period)

        method_scores[method] = score
        if significant:
            significant_methods.append(method)

    return {
        "period": period,
        "method_scores": method_scores,
        "significant_methods": significant_methods,
        "num_significant": len(significant_methods),
        "combined_score": ensemble_result.combined_scores.get(period, 0.0),
        "agreement": len(significant_methods) / len(ensemble_result.results) if ensemble_result.results else 0,
    }
