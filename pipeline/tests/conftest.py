"""Pytest fixtures for Seasonality Analyzer tests."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile

from tests.data.synthetic import generate_seasonal_data


@pytest.fixture
def sample_seasonal_data():
    """Generate sample seasonal data for testing."""
    return generate_seasonal_data(
        n_days=365,
        periods=[7, 30],
        trend="linear",
        noise_level=0.1,
        outlier_rate=0.01,
        missing_rate=0.005,
    )


@pytest.fixture
def sample_data_no_seasonality():
    """Generate data without clear seasonality."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=365, freq="D")
    values = np.random.randn(365) * 10 + 100  # Pure noise
    return pd.DataFrame({
        "timestamp": dates,
        "value": values,
    }).set_index("timestamp")


@pytest.fixture
def sample_multi_sensor_data():
    """Generate multi-sensor data."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=365, freq="D")

    df = pd.DataFrame(index=dates)
    df.index.name = "timestamp"

    # Sensor with strong weekly seasonality
    t = np.arange(365)
    df["sensor_1"] = 100 + 10 * np.sin(2 * np.pi * t / 7) + np.random.randn(365)

    # Sensor with weak seasonality
    df["sensor_2"] = 50 + 2 * np.sin(2 * np.pi * t / 30) + np.random.randn(365) * 5

    # Sensor with no seasonality
    df["sensor_3"] = np.random.randn(365) * 10 + 75

    return df


@pytest.fixture
def temp_csv_file(sample_multi_sensor_data):
    """Create temporary CSV file with sample data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        # Reset index and add date/time columns
        df = sample_multi_sensor_data.reset_index()
        df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
        df["time"] = df["timestamp"].dt.strftime("%H:%M:%S")
        df = df.drop(columns=["timestamp"])
        df.to_csv(f.name, index=False)
        yield Path(f.name)

    # Cleanup
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def default_config():
    """Get default configuration."""
    from seasonality.config import SeasonalityConfig
    return SeasonalityConfig()


@pytest.fixture
def strict_config():
    """Get strict mode configuration."""
    from seasonality.config import SeasonalityConfig, ScoringConfig
    return SeasonalityConfig(
        scoring=ScoringConfig(mode="strict")
    )


@pytest.fixture
def exploratory_config():
    """Get exploratory mode configuration."""
    from seasonality.config import SeasonalityConfig, ScoringConfig
    return SeasonalityConfig(
        scoring=ScoringConfig(mode="exploratory")
    )
