"""季節性分析のランキングと分類。"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from loguru import logger

from seasonality.detection.ensemble import EnsembleResult
from seasonality.scoring.metrics import aggregate_method_scores, calculate_composite_score
from seasonality.config import ScoringConfig


def classify_seasonality(
    scores: Dict[str, float],
    config: Optional[ScoringConfig] = None,
) -> str:
    """季節性の信頼度レベルを分類します。

    Args:
        scores: 手法スコアを含む辞書
        config: 閾値を含むスコアリング設定

    Returns:
        信頼度ラベル: 'High', 'Medium', 'Low', または 'None'
    """
    config = config or ScoringConfig()
    mode = config.mode

    stl_max = scores.get("stl_max", 0.0)
    acf_max = scores.get("acf_max", 0.0)
    fourier_max = scores.get("fourier_max", 0.0)

    if mode == "strict":
        thresholds = config.strict_thresholds

        stl_ok = stl_max > thresholds.stl_strength
        acf_ok = acf_max > thresholds.acf_peak
        fourier_ok = fourier_max > thresholds.delta_r2

        if stl_ok and acf_ok and fourier_ok:
            return "High"
        elif (stl_ok and acf_ok) or (stl_ok and fourier_ok) or (acf_ok and fourier_ok):
            return "Medium"
        elif stl_ok or acf_ok or fourier_ok:
            return "Low"
        else:
            return "None"

    else:  # exploratory mode
        thresholds = config.exploratory_thresholds

        has_signal = (
            stl_max > thresholds.stl_strength or
            acf_max > thresholds.acf_peak or
            fourier_max > thresholds.delta_r2
        )

        has_strong_signal = (
            stl_max > config.strict_thresholds.stl_strength or
            acf_max > config.strict_thresholds.acf_peak or
            fourier_max > config.strict_thresholds.delta_r2
        )

        has_weak_signal = (
            stl_max > thresholds.stl_strength * 0.5 or
            acf_max > thresholds.acf_peak * 0.5 or
            fourier_max > thresholds.delta_r2 * 0.5
        )

        if has_signal and has_strong_signal:
            return "High"
        elif has_signal:
            return "Medium"
        elif has_weak_signal:
            return "Low"
        else:
            return "None"


def create_scores_dataframe(
    ensemble_results: List[EnsembleResult],
    config: Optional[ScoringConfig] = None,
) -> pd.DataFrame:
    """アンサンブル結果から全スコアを含むDataFrameを作成します。

    Args:
        ensemble_results: アンサンブル検出結果のリスト
        config: スコアリング設定

    Returns:
        センサーでインデックス化されたスコアを含むDataFrame
    """
    config = config or ScoringConfig()
    all_scores = []

    for result in ensemble_results:
        scores = aggregate_method_scores(result)
        scores["composite_score"] = calculate_composite_score(scores, config)
        scores["confidence_strict"] = classify_seasonality(
            scores, ScoringConfig(mode="strict")
        )
        scores["confidence_exploratory"] = classify_seasonality(
            scores, ScoringConfig(mode="exploratory")
        )
        all_scores.append(scores)

    df = pd.DataFrame(all_scores)

    if "sensor" in df.columns:
        df = df.set_index("sensor")

    return df


def generate_ranking(
    scores_df: pd.DataFrame,
    sort_by: str = "composite_score",
    ascending: bool = False,
) -> pd.DataFrame:
    """ランク付けされたDataFrameを生成します。

    Args:
        scores_df: スコアを含むDataFrame
        sort_by: ソート基準とする列
        ascending: ソート順

    Returns:
        ランク列を含むソート済みDataFrame
    """
    ranking = scores_df.sort_values(sort_by, ascending=ascending).copy()
    ranking["rank"] = range(1, len(ranking) + 1)

    # ランクを最初の列に配置
    cols = ["rank"] + [c for c in ranking.columns if c != "rank"]
    ranking = ranking[cols]

    return ranking


def get_top_sensors(
    scores_df: pd.DataFrame,
    n: int = 10,
    min_confidence: Literal["High", "Medium", "Low", "None"] = "Low",
    mode: Literal["strict", "exploratory"] = "strict",
) -> pd.DataFrame:
    """総合スコアで上位N個のセンサーを取得します。

    Args:
        scores_df: スコアを含むDataFrame
        n: 返却する上位センサーの数
        min_confidence: 含める最小信頼度レベル
        mode: 使用する信頼度列

    Returns:
        上位センサーを含むDataFrame
    """
    confidence_col = f"confidence_{mode}"
    confidence_order = {"High": 3, "Medium": 2, "Low": 1, "None": 0}

    if confidence_col not in scores_df.columns:
        logger.warning(f"信頼度列 {confidence_col} が見つかりません")
        return scores_df.nlargest(n, "composite_score")

    min_level = confidence_order.get(min_confidence, 0)

    # 最小信頼度でフィルタリング
    filtered = scores_df[
        scores_df[confidence_col].map(confidence_order) >= min_level
    ]

    return filtered.nlargest(n, "composite_score")


def summarize_confidence_distribution(
    scores_df: pd.DataFrame,
    mode: Literal["strict", "exploratory"] = "strict",
) -> Dict[str, int]:
    """信頼度レベルの分布を要約します。

    Args:
        scores_df: スコアを含むDataFrame
        mode: 使用する信頼度列

    Returns:
        信頼度レベルをカウントにマッピングした辞書
    """
    confidence_col = f"confidence_{mode}"

    if confidence_col not in scores_df.columns:
        return {}

    return scores_df[confidence_col].value_counts().to_dict()


def filter_by_period(
    scores_df: pd.DataFrame,
    period: int,
    min_score: float = 0.0,
) -> pd.DataFrame:
    """特定周期で有意な検出があるセンサーをフィルタリングします。

    Args:
        scores_df: スコアを含むDataFrame
        period: 対象周期
        min_score: 最小統合スコア

    Returns:
        フィルタリングされたDataFrame
    """
    period_col = f"combined_{period}"

    if period_col not in scores_df.columns:
        logger.warning(f"周期列 {period_col} が見つかりません")
        return scores_df

    return scores_df[scores_df[period_col] >= min_score]


def generate_summary_statistics(
    scores_df: pd.DataFrame,
    config: Optional[ScoringConfig] = None,
) -> Dict[str, Any]:
    """分析のサマリー統計を生成します。

    Args:
        scores_df: スコアを含むDataFrame
        config: スコアリング設定

    Returns:
        サマリー統計を含む辞書
    """
    config = config or ScoringConfig()

    summary = {
        "total_sensors": len(scores_df),
        "confidence_distribution_strict": summarize_confidence_distribution(scores_df, "strict"),
        "confidence_distribution_exploratory": summarize_confidence_distribution(scores_df, "exploratory"),
    }

    # スコア統計
    if "composite_score" in scores_df.columns:
        summary["composite_score_stats"] = {
            "mean": float(scores_df["composite_score"].mean()),
            "std": float(scores_df["composite_score"].std()),
            "min": float(scores_df["composite_score"].min()),
            "max": float(scores_df["composite_score"].max()),
            "median": float(scores_df["composite_score"].median()),
        }

    # 手法固有の統計
    for method in ["stl", "acf", "fourier", "periodogram"]:
        col = f"{method}_max"
        if col in scores_df.columns:
            summary[f"{method}_stats"] = {
                "mean": float(scores_df[col].mean()),
                "std": float(scores_df[col].std()),
                "max": float(scores_df[col].max()),
            }

    # 最良周期の分布
    if "best_period" in scores_df.columns:
        summary["best_period_distribution"] = (
            scores_df["best_period"].value_counts().to_dict()
        )

    return summary
