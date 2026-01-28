"""Scoring metrics for seasonality analysis."""

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
    """Normalize a score to 0-1 range.

    Args:
        value: Input value.
        min_val: Minimum expected value.
        max_val: Maximum expected value.

    Returns:
        Normalized value between 0 and 1.
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
    """Aggregate scores from ensemble result into flat dictionary.

    Args:
        ensemble_result: Result from ensemble detection.
        periods: List of periods to include. If None, uses all.

    Returns:
        Dictionary with aggregated scores.
    """
    scores = {"sensor": ensemble_result.sensor}

    # Get periods from results if not specified
    if periods is None:
        all_periods = set()
        for result in ensemble_result.results.values():
            all_periods.update(result.periods.keys())
        periods = sorted(all_periods)

    # Extract scores for each method and period
    for method, result in ensemble_result.results.items():
        max_score = 0.0

        for period in periods:
            period_score = result.get_score(period)
            key = f"{method}_{period}"
            scores[key] = period_score
            max_score = max(max_score, period_score)

        scores[f"{method}_max"] = max_score

    # Add combined scores
    for period in periods:
        scores[f"combined_{period}"] = ensemble_result.combined_scores.get(period, 0.0)

    scores["best_period"] = ensemble_result.best_period
    scores["max_combined_score"] = ensemble_result.max_combined_score

    return scores


def calculate_composite_score(
    scores: Dict[str, float],
    config: Optional[ScoringConfig] = None,
) -> float:
    """Calculate weighted composite score from method scores.

    Args:
        scores: Dictionary with method scores (e.g., 'stl_max', 'acf_max', 'fourier_full').
        config: Scoring configuration with weights.

    Returns:
        Weighted composite score.
    """
    if config is None:
        weights = {"stl": 0.3, "acf": 0.3, "fourier": 0.4}
    else:
        weights = config.weights

    stl_score = scores.get("stl_max", 0.0)
    acf_score = scores.get("acf_max", 0.0)
    fourier_score = scores.get("fourier_max", 0.0)

    # Normalize fourier score (delta R² is typically smaller)
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
    """Calculate all scores for a single sensor.

    Args:
        ensemble_result: Result from ensemble detection.
        config: Scoring configuration.

    Returns:
        Dictionary with all calculated scores.
    """
    config = config or ScoringConfig()

    # Get aggregated method scores
    scores = aggregate_method_scores(ensemble_result)

    # Calculate composite score
    scores["composite_score"] = calculate_composite_score(scores, config)

    return scores


def calculate_period_confidence(
    ensemble_result: EnsembleResult,
    period: int,
) -> Dict[str, Any]:
    """Calculate confidence metrics for a specific period.

    Args:
        ensemble_result: Ensemble detection result.
        period: Target period.

    Returns:
        Dictionary with period-specific confidence metrics.
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
