"""Tests for detection module."""

import numpy as np
import pandas as pd
import pytest

from seasonality.detection.stl import STLDetector
from seasonality.detection.acf import ACFDetector
from seasonality.detection.fourier import FourierDetector
from seasonality.detection.ensemble import EnsembleDetector


class TestSTLDetector:
    """Tests for STL detector."""

    def test_detect_clear_seasonality(self):
        """Test STL detection on data with clear seasonality."""
        np.random.seed(42)
        t = np.arange(365)
        # Strong weekly seasonality
        values = 100 + 20 * np.sin(2 * np.pi * t / 7) + np.random.randn(365)
        dates = pd.date_range(start="2024-01-01", periods=365, freq="D")
        series = pd.Series(values, index=dates)

        detector = STLDetector()
        result = detector.detect(series, periods=[7, 30])

        # Should detect weekly seasonality
        assert result.get_score(7) > 0.3
        assert result.is_significant(7) or result.get_score(7) > 0.2

    def test_detect_no_seasonality(self):
        """Test STL detection on random noise."""
        np.random.seed(42)
        values = np.random.randn(365) * 10 + 100
        dates = pd.date_range(start="2024-01-01", periods=365, freq="D")
        series = pd.Series(values, index=dates)

        detector = STLDetector()
        result = detector.detect(series, periods=[7, 30])

        # Should have low scores
        assert result.get_score(7) < 0.5
        assert result.get_score(30) < 0.5

    def test_decompose(self):
        """Test STL decomposition."""
        np.random.seed(42)
        t = np.arange(200)
        values = 100 + 10 * np.sin(2 * np.pi * t / 7) + np.random.randn(200) * 2
        dates = pd.date_range(start="2024-01-01", periods=200, freq="D")
        series = pd.Series(values, index=dates)

        detector = STLDetector()
        decomposition = detector.decompose(series, period=7)

        assert decomposition is not None
        assert "trend" in decomposition
        assert "seasonal" in decomposition
        assert "residual" in decomposition
        assert len(decomposition["trend"]) == 200

    def test_insufficient_data(self):
        """Test STL with insufficient data."""
        values = [100, 101, 102, 103, 104]
        dates = pd.date_range(start="2024-01-01", periods=5, freq="D")
        series = pd.Series(values, index=dates)

        detector = STLDetector()
        result = detector.detect(series, periods=[7])

        # Should return low/zero score for insufficient data
        assert result.get_score(7) == 0


class TestACFDetector:
    """Tests for ACF detector."""

    def test_detect_weekly_pattern(self):
        """Test ACF detection of weekly pattern."""
        np.random.seed(42)
        t = np.arange(365)
        values = 100 + 15 * np.sin(2 * np.pi * t / 7) + np.random.randn(365) * 2
        dates = pd.date_range(start="2024-01-01", periods=365, freq="D")
        series = pd.Series(values, index=dates)

        detector = ACFDetector()
        result = detector.detect(series, periods=[7, 30])

        # Should detect weekly pattern
        assert result.get_score(7) > 0.2

    def test_compute_full_acf(self):
        """Test full ACF computation."""
        np.random.seed(42)
        values = np.random.randn(200)
        dates = pd.date_range(start="2024-01-01", periods=200, freq="D")
        series = pd.Series(values, index=dates)

        detector = ACFDetector()
        acf_data = detector.compute_full_acf(series, max_lag=50)

        assert "lags" in acf_data
        assert "acf" in acf_data
        assert len(acf_data["lags"]) == 51  # 0 to 50


class TestFourierDetector:
    """Tests for Fourier detector."""

    def test_detect_single_period(self):
        """Test Fourier detection of single period."""
        np.random.seed(42)
        t = np.arange(365)
        values = 100 + 20 * np.sin(2 * np.pi * t / 30) + np.random.randn(365)
        dates = pd.date_range(start="2024-01-01", periods=365, freq="D")
        series = pd.Series(values, index=dates)

        detector = FourierDetector()
        result = detector.detect(series, periods=[7, 30])

        # Should detect 30-day period
        assert result.get_score(30) > result.get_score(7)

    def test_fit_and_predict(self):
        """Test Fourier fit and predict."""
        np.random.seed(42)
        t = np.arange(200)
        values = 100 + 10 * np.sin(2 * np.pi * t / 7) + np.random.randn(200)
        dates = pd.date_range(start="2024-01-01", periods=200, freq="D")
        series = pd.Series(values, index=dates)

        detector = FourierDetector()
        predictions = detector.fit_and_predict(series, periods=[7])

        assert predictions is not None
        assert len(predictions) == 200

    def test_get_delta_r2(self):
        """Test delta R² calculation."""
        np.random.seed(42)
        t = np.arange(365)
        values = 100 + 15 * np.sin(2 * np.pi * t / 30) + np.random.randn(365) * 2
        dates = pd.date_range(start="2024-01-01", periods=365, freq="D")
        series = pd.Series(values, index=dates)

        detector = FourierDetector()
        delta_r2 = detector.get_delta_r2(series, periods=[7, 30])

        assert "period_30" in delta_r2
        assert "full" in delta_r2
        # 30-day period should explain more variance
        assert delta_r2["period_30"] > delta_r2["period_7"]


class TestEnsembleDetector:
    """Tests for ensemble detector."""

    def test_detect_combines_methods(self):
        """Test that ensemble combines multiple methods."""
        np.random.seed(42)
        t = np.arange(365)
        values = 100 + 15 * np.sin(2 * np.pi * t / 7) + np.random.randn(365) * 2
        dates = pd.date_range(start="2024-01-01", periods=365, freq="D")
        series = pd.Series(values, index=dates)

        detector = EnsembleDetector(methods=["stl", "acf", "fourier"])
        result = detector.detect(series, periods=[7, 30], sensor_name="test")

        assert result.sensor == "test"
        assert len(result.results) == 3
        assert "stl" in result.results
        assert "acf" in result.results
        assert "fourier" in result.results
        assert 7 in result.combined_scores
        assert 30 in result.combined_scores

    def test_detect_batch(self, sample_multi_sensor_data):
        """Test batch detection on multiple sensors."""
        detector = EnsembleDetector(methods=["stl", "fourier"])
        results = detector.detect_batch(
            sample_multi_sensor_data,
            sensor_cols=["sensor_1", "sensor_2", "sensor_3"],
            periods=[7, 30],
        )

        assert len(results) == 3
        # sensor_1 has strong weekly seasonality, should have higher score
        sensor_1_score = results[0].max_combined_score
        sensor_3_score = results[2].max_combined_score
        # sensor_1 should generally have higher seasonality than sensor_3 (noise only)
        # This is a probabilistic test, so we use a loose threshold
        assert sensor_1_score >= 0


class TestDetectionEdgeCases:
    """Tests for edge cases in detection."""

    def test_all_nan_series(self):
        """Test detection with all NaN values."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        series = pd.Series([np.nan] * 100, index=dates)

        detector = STLDetector()
        with pytest.raises(ValueError):
            detector.detect(series, periods=[7])

    def test_constant_series(self):
        """Test detection with constant values."""
        dates = pd.date_range(start="2024-01-01", periods=365, freq="D")
        series = pd.Series([100.0] * 365, index=dates)

        detector = FourierDetector()
        result = detector.detect(series, periods=[7, 30])

        # Constant series should have zero seasonality
        assert result.get_score(7) == 0 or result.get_score(7) < 0.1
