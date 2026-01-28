"""Ranking and classification for seasonality analysis."""

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
    """Classify seasonality confidence level.

    Args:
        scores: Dictionary with method scores.
        config: Scoring configuration with thresholds.

    Returns:
        Confidence label: 'High', 'Medium', 'Low', or 'None'.
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
    """Create DataFrame with all scores from ensemble results.

    Args:
        ensemble_results: List of ensemble detection results.
        config: Scoring configuration.

    Returns:
        DataFrame with scores indexed by sensor.
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
    """Generate ranked DataFrame.

    Args:
        scores_df: DataFrame with scores.
        sort_by: Column to sort by.
        ascending: Sort order.

    Returns:
        Sorted DataFrame with rank column.
    """
    ranking = scores_df.sort_values(sort_by, ascending=ascending).copy()
    ranking["rank"] = range(1, len(ranking) + 1)

    # Reorder columns to put rank first
    cols = ["rank"] + [c for c in ranking.columns if c != "rank"]
    ranking = ranking[cols]

    return ranking


def get_top_sensors(
    scores_df: pd.DataFrame,
    n: int = 10,
    min_confidence: Literal["High", "Medium", "Low", "None"] = "Low",
    mode: Literal["strict", "exploratory"] = "strict",
) -> pd.DataFrame:
    """Get top N sensors by composite score.

    Args:
        scores_df: DataFrame with scores.
        n: Number of top sensors to return.
        min_confidence: Minimum confidence level to include.
        mode: Which confidence column to use.

    Returns:
        DataFrame with top sensors.
    """
    confidence_col = f"confidence_{mode}"
    confidence_order = {"High": 3, "Medium": 2, "Low": 1, "None": 0}

    if confidence_col not in scores_df.columns:
        logger.warning(f"Confidence column {confidence_col} not found")
        return scores_df.nlargest(n, "composite_score")

    min_level = confidence_order.get(min_confidence, 0)

    # Filter by minimum confidence
    filtered = scores_df[
        scores_df[confidence_col].map(confidence_order) >= min_level
    ]

    return filtered.nlargest(n, "composite_score")


def summarize_confidence_distribution(
    scores_df: pd.DataFrame,
    mode: Literal["strict", "exploratory"] = "strict",
) -> Dict[str, int]:
    """Summarize distribution of confidence levels.

    Args:
        scores_df: DataFrame with scores.
        mode: Which confidence column to use.

    Returns:
        Dictionary mapping confidence level to count.
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
    """Filter sensors with significant detection at specific period.

    Args:
        scores_df: DataFrame with scores.
        period: Target period.
        min_score: Minimum combined score.

    Returns:
        Filtered DataFrame.
    """
    period_col = f"combined_{period}"

    if period_col not in scores_df.columns:
        logger.warning(f"Period column {period_col} not found")
        return scores_df

    return scores_df[scores_df[period_col] >= min_score]


def generate_summary_statistics(
    scores_df: pd.DataFrame,
    config: Optional[ScoringConfig] = None,
) -> Dict[str, Any]:
    """Generate summary statistics for the analysis.

    Args:
        scores_df: DataFrame with scores.
        config: Scoring configuration.

    Returns:
        Dictionary with summary statistics.
    """
    config = config or ScoringConfig()

    summary = {
        "total_sensors": len(scores_df),
        "confidence_distribution_strict": summarize_confidence_distribution(scores_df, "strict"),
        "confidence_distribution_exploratory": summarize_confidence_distribution(scores_df, "exploratory"),
    }

    # Score statistics
    if "composite_score" in scores_df.columns:
        summary["composite_score_stats"] = {
            "mean": float(scores_df["composite_score"].mean()),
            "std": float(scores_df["composite_score"].std()),
            "min": float(scores_df["composite_score"].min()),
            "max": float(scores_df["composite_score"].max()),
            "median": float(scores_df["composite_score"].median()),
        }

    # Method-specific statistics
    for method in ["stl", "acf", "fourier", "periodogram"]:
        col = f"{method}_max"
        if col in scores_df.columns:
            summary[f"{method}_stats"] = {
                "mean": float(scores_df[col].mean()),
                "std": float(scores_df[col].std()),
                "max": float(scores_df[col].max()),
            }

    # Best period distribution
    if "best_period" in scores_df.columns:
        summary["best_period_distribution"] = (
            scores_df["best_period"].value_counts().to_dict()
        )

    return summary
