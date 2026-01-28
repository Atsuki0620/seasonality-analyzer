"""Tests for scoring module."""

import numpy as np
import pandas as pd
import pytest

from seasonality.scoring.metrics import (
    normalize_score,
    calculate_composite_score,
    aggregate_method_scores,
)
from seasonality.scoring.ranking import (
    classify_seasonality,
    create_scores_dataframe,
    generate_ranking,
    summarize_confidence_distribution,
)
from seasonality.config import ScoringConfig


class TestMetrics:
    """Tests for scoring metrics."""

    def test_normalize_score(self):
        """Test score normalization."""
        assert normalize_score(0.5, 0, 1) == pytest.approx(0.5)
        assert normalize_score(0, 0, 1) == pytest.approx(0)
        assert normalize_score(1, 0, 1) == pytest.approx(1)
        assert normalize_score(2, 0, 1) == pytest.approx(1)  # Clipped
        assert normalize_score(-1, 0, 1) == pytest.approx(0)  # Clipped

    def test_normalize_score_nan(self):
        """Test normalization of NaN values."""
        assert normalize_score(np.nan) == 0

    def test_calculate_composite_score(self):
        """Test composite score calculation."""
        scores = {
            "stl_max": 0.6,
            "acf_max": 0.4,
            "fourier_max": 0.1,
        }

        composite = calculate_composite_score(scores)

        # Default weights: stl=0.3, acf=0.3, fourier=0.4
        # 0.3*0.6 + 0.3*0.4 + 0.4*min(0.1*10, 1)
        expected = 0.3 * 0.6 + 0.3 * 0.4 + 0.4 * 1.0
        assert composite == pytest.approx(expected, rel=0.01)

    def test_calculate_composite_score_custom_weights(self):
        """Test composite score with custom weights."""
        scores = {
            "stl_max": 0.5,
            "acf_max": 0.5,
            "fourier_max": 0.05,
        }
        config = ScoringConfig(
            weights={"stl": 0.5, "acf": 0.5, "fourier": 0.0}
        )

        composite = calculate_composite_score(scores, config)

        expected = 0.5 * 0.5 + 0.5 * 0.5
        assert composite == pytest.approx(expected, rel=0.01)


class TestClassification:
    """Tests for seasonality classification."""

    def test_classify_high_strict(self):
        """Test High classification in strict mode."""
        scores = {
            "stl_max": 0.6,
            "acf_max": 0.4,
            "fourier_max": 0.15,
        }
        config = ScoringConfig(mode="strict")

        result = classify_seasonality(scores, config)

        assert result == "High"

    def test_classify_medium_strict(self):
        """Test Medium classification in strict mode."""
        scores = {
            "stl_max": 0.6,
            "acf_max": 0.4,
            "fourier_max": 0.05,  # Below threshold
        }
        config = ScoringConfig(mode="strict")

        result = classify_seasonality(scores, config)

        assert result == "Medium"

    def test_classify_low_strict(self):
        """Test Low classification in strict mode."""
        scores = {
            "stl_max": 0.6,
            "acf_max": 0.2,  # Below threshold
            "fourier_max": 0.05,  # Below threshold
        }
        config = ScoringConfig(mode="strict")

        result = classify_seasonality(scores, config)

        assert result == "Low"

    def test_classify_none_strict(self):
        """Test None classification in strict mode."""
        scores = {
            "stl_max": 0.3,  # Below threshold
            "acf_max": 0.1,  # Below threshold
            "fourier_max": 0.01,  # Below threshold
        }
        config = ScoringConfig(mode="strict")

        result = classify_seasonality(scores, config)

        assert result == "None"

    def test_classify_exploratory_mode(self):
        """Test classification in exploratory mode."""
        scores = {
            "stl_max": 0.3,  # Above exploratory threshold
            "acf_max": 0.2,  # Above exploratory threshold
            "fourier_max": 0.03,
        }
        config = ScoringConfig(mode="exploratory")

        result = classify_seasonality(scores, config)

        # Should classify as Medium or High in exploratory mode
        assert result in ["High", "Medium"]


class TestRanking:
    """Tests for ranking functions."""

    def test_generate_ranking(self):
        """Test ranking generation."""
        data = {
            "composite_score": [0.8, 0.5, 0.9, 0.3],
            "stl_max": [0.7, 0.4, 0.8, 0.2],
        }
        df = pd.DataFrame(data, index=["s1", "s2", "s3", "s4"])

        ranking = generate_ranking(df, sort_by="composite_score")

        assert ranking.iloc[0].name == "s3"  # Highest score
        assert ranking.iloc[1].name == "s1"
        assert ranking.iloc[-1].name == "s4"  # Lowest score
        assert "rank" in ranking.columns
        assert ranking.iloc[0]["rank"] == 1

    def test_summarize_confidence_distribution(self):
        """Test confidence distribution summary."""
        data = {
            "confidence_strict": ["High", "High", "Medium", "Low", "None"],
        }
        df = pd.DataFrame(data, index=["s1", "s2", "s3", "s4", "s5"])

        dist = summarize_confidence_distribution(df, mode="strict")

        assert dist["High"] == 2
        assert dist["Medium"] == 1
        assert dist["Low"] == 1
        assert dist["None"] == 1


class TestIntegration:
    """Integration tests for scoring pipeline."""

    def test_full_scoring_pipeline(self, sample_multi_sensor_data):
        """Test full scoring pipeline."""
        from seasonality.detection.ensemble import EnsembleDetector

        # Run detection
        detector = EnsembleDetector(methods=["stl", "fourier"])
        ensemble_results = detector.detect_batch(
            sample_multi_sensor_data,
            sensor_cols=["sensor_1", "sensor_2", "sensor_3"],
            periods=[7, 30],
        )

        # Create scores dataframe
        scores_df = create_scores_dataframe(ensemble_results)

        assert len(scores_df) == 3
        assert "composite_score" in scores_df.columns
        assert "confidence_strict" in scores_df.columns
        assert "confidence_exploratory" in scores_df.columns

        # Generate ranking
        ranking = generate_ranking(scores_df)

        assert "rank" in ranking.columns
        assert ranking.iloc[0]["rank"] == 1
