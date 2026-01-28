"""Tests for preprocessing module."""

import numpy as np
import pandas as pd
import pytest

from seasonality.preprocessing.resample import (
    resample_to_daily,
    resample_and_interpolate,
)
from seasonality.preprocessing.outlier import (
    detect_outliers_iqr,
    detect_outliers_zscore,
    detect_outliers,
    remove_outliers,
)
from seasonality.preprocessing.changepoint import (
    detect_changepoints,
    segment_by_changepoints,
)
from seasonality.preprocessing.impute import (
    interpolate_missing,
    get_missing_report,
)


class TestResample:
    """Tests for resampling functions."""

    def test_resample_to_daily_mean(self):
        """Test daily resampling with mean aggregation."""
        # Create hourly data
        dates = pd.date_range(start="2024-01-01", periods=48, freq="h")
        values = np.arange(48)
        series = pd.Series(values, index=dates)

        result = resample_to_daily(series, method="mean")

        assert len(result) == 2
        assert result.iloc[0] == pytest.approx(11.5, rel=0.01)  # Mean of 0-23
        assert result.iloc[1] == pytest.approx(35.5, rel=0.01)  # Mean of 24-47

    def test_resample_to_daily_median(self):
        """Test daily resampling with median aggregation."""
        dates = pd.date_range(start="2024-01-01", periods=48, freq="h")
        values = np.arange(48)
        series = pd.Series(values, index=dates)

        result = resample_to_daily(series, method="median")

        assert len(result) == 2

    def test_resample_and_interpolate_linear(self):
        """Test resampling with linear interpolation."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        values = [1, np.nan, 3, np.nan, 5, 6, np.nan, 8, 9, 10]
        series = pd.Series(values, index=dates)

        result = resample_and_interpolate(
            series,
            resample_freq="D",
            resample_method="mean",
            interp_method="linear",
        )

        # Check that NaNs are filled
        assert result.isna().sum() == 0

    def test_resample_no_interpolation(self):
        """Test resampling without interpolation."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        values = [1, np.nan, 3, np.nan, 5, 6, np.nan, 8, 9, 10]
        series = pd.Series(values, index=dates)

        result = resample_and_interpolate(
            series,
            resample_freq="D",
            resample_method="mean",
            interp_method=None,
        )

        # NaNs should remain
        assert result.isna().sum() == 3


class TestOutlierDetection:
    """Tests for outlier detection functions."""

    def test_detect_outliers_iqr(self):
        """Test IQR-based outlier detection."""
        # Normal values with one outlier
        values = [10, 11, 10, 12, 11, 10, 50, 11, 10, 12]
        series = pd.Series(values)

        outliers = detect_outliers_iqr(series, k=1.5)

        assert outliers.sum() == 1
        assert outliers.iloc[6] == True  # The outlier at index 6

    def test_detect_outliers_zscore(self):
        """Test Z-score-based outlier detection."""
        np.random.seed(42)
        values = np.random.randn(100) * 10 + 100
        values[50] = 200  # Add outlier

        series = pd.Series(values)
        outliers = detect_outliers_zscore(series, threshold=3)

        assert outliers.sum() >= 1
        assert outliers.iloc[50] == True

    def test_detect_outliers_with_method(self):
        """Test outlier detection with method parameter."""
        values = [10, 11, 10, 12, 11, 10, 50, 11, 10, 12]
        series = pd.Series(values)

        iqr_outliers = detect_outliers(series, method="iqr")
        zscore_outliers = detect_outliers(series, method="zscore")

        # Both should detect the obvious outlier
        assert iqr_outliers.sum() >= 1

    def test_remove_outliers_nan(self):
        """Test removing outliers by replacing with NaN."""
        values = [10, 11, 10, 12, 11, 10, 50, 11, 10, 12]
        series = pd.Series(values)

        result = remove_outliers(series, method="iqr", replace_with="nan")

        assert pd.isna(result.iloc[6])

    def test_remove_outliers_median(self):
        """Test removing outliers by replacing with median."""
        values = [10, 11, 10, 12, 11, 10, 50, 11, 10, 12]
        series = pd.Series(values)

        result = remove_outliers(series, method="iqr", replace_with="median")

        # Outlier should be replaced with median
        median_val = pd.Series(values).median()
        assert result.iloc[6] == pytest.approx(median_val, rel=0.01)


class TestChangepoint:
    """Tests for changepoint detection."""

    def test_detect_changepoints_level_shift(self):
        """Test detecting level shift changepoint."""
        np.random.seed(42)
        # Data with clear level shift
        values = np.concatenate([
            np.random.randn(100) + 10,  # First segment
            np.random.randn(100) + 20,  # Second segment (shifted)
        ])
        dates = pd.date_range(start="2024-01-01", periods=200, freq="D")
        series = pd.Series(values, index=dates)

        changepoints = detect_changepoints(series, method="pelt", penalty=5)

        # Should detect at least one changepoint
        assert len(changepoints) >= 1

    def test_segment_by_changepoints(self):
        """Test segmenting series by changepoints."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        values = np.arange(100)
        series = pd.Series(values, index=dates)

        # Create two segments
        changepoints = [dates[50]]
        segments = segment_by_changepoints(series, changepoints)

        assert len(segments) == 2
        assert segments[0].segment_number == 1
        assert segments[1].segment_number == 2


class TestImpute:
    """Tests for imputation functions."""

    def test_interpolate_linear(self):
        """Test linear interpolation."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        values = [1.0, np.nan, 3.0, np.nan, 5.0, 6.0, np.nan, 8.0, 9.0, 10.0]
        series = pd.Series(values, index=dates)

        result = interpolate_missing(series, method="linear")

        assert result.isna().sum() == 0
        assert result.iloc[1] == pytest.approx(2.0, rel=0.01)
        assert result.iloc[3] == pytest.approx(4.0, rel=0.01)

    def test_get_missing_report(self):
        """Test missing value report generation."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        values = [1.0, np.nan, np.nan, np.nan, 5.0, 6.0, np.nan, 8.0, 9.0, 10.0]
        series = pd.Series(values, index=dates)

        report = get_missing_report(series)

        assert report["total_count"] == 10
        assert report["missing_count"] == 4
        assert report["missing_rate"] == pytest.approx(0.4, rel=0.01)
        assert report["max_gap"] == 3  # Three consecutive NaNs
