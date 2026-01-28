"""Configuration models for Seasonality Analyzer using Pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    """Configuration for data loading."""

    date_column: str = Field(default="date", description="Column name for date")
    time_column: Optional[str] = Field(default="time", description="Column name for time (optional)")
    sensor_columns: Optional[List[str]] = Field(
        default=None,
        description="List of sensor column names. If None, auto-detect columns starting with 'sensor_'",
    )
    sensor_prefix: str = Field(default="sensor_", description="Prefix for auto-detecting sensor columns")


class PreprocessingConfig(BaseModel):
    """Configuration for preprocessing pipeline."""

    resample_freq: str = Field(default="D", description="Resampling frequency (D=daily, W=weekly, etc.)")
    resample_method: Literal["mean", "median", "ffill"] = Field(
        default="mean", description="Aggregation method for resampling"
    )
    outlier_method: Literal["iqr", "zscore", "iforest", "none"] = Field(
        default="iqr", description="Outlier detection method"
    )
    outlier_threshold: float = Field(default=1.5, description="Threshold for outlier detection (k for IQR, z for zscore)")
    impute_method: Literal["linear", "spline", "ffill", "none"] = Field(
        default="linear", description="Interpolation method for missing values"
    )
    changepoint_method: Literal["pelt", "window", "none"] = Field(
        default="none", description="Changepoint detection method"
    )
    changepoint_penalty: float = Field(default=10.0, description="Penalty parameter for changepoint detection")
    min_segment_size: int = Field(default=30, description="Minimum segment size for changepoint detection")


class DetectionConfig(BaseModel):
    """Configuration for seasonality detection."""

    periods: List[int] = Field(default=[7, 30, 365], description="Target periods to detect (in days)")
    methods: List[Literal["stl", "acf", "periodogram", "fourier"]] = Field(
        default=["stl", "acf", "periodogram", "fourier"],
        description="Detection methods to use",
    )
    stl_seasonal: int = Field(default=7, description="Seasonal smoother for STL decomposition")
    stl_robust: bool = Field(default=True, description="Use robust fitting for STL")
    acf_max_lag: int = Field(default=400, description="Maximum lag for ACF calculation")
    acf_tolerance: int = Field(default=3, description="Tolerance for peak detection around target periods")
    periodogram_min_period: int = Field(default=2, description="Minimum period for Lomb-Scargle")
    periodogram_max_period: int = Field(default=500, description="Maximum period for Lomb-Scargle")
    fap_threshold: float = Field(default=0.05, description="False Alarm Probability threshold for significance")


class StrictThresholds(BaseModel):
    """Thresholds for strict mode scoring."""

    stl_strength: float = Field(default=0.5, description="Minimum STL seasonal strength")
    acf_peak: float = Field(default=0.3, description="Minimum ACF peak value")
    delta_r2: float = Field(default=0.1, description="Minimum Fourier delta R²")


class ExploratoryThresholds(BaseModel):
    """Thresholds for exploratory mode scoring."""

    stl_strength: float = Field(default=0.2, description="Minimum STL seasonal strength")
    acf_peak: float = Field(default=0.15, description="Minimum ACF peak value")
    delta_r2: float = Field(default=0.05, description="Minimum Fourier delta R²")


class ScoringConfig(BaseModel):
    """Configuration for scoring and classification."""

    mode: Literal["strict", "exploratory"] = Field(default="strict", description="Scoring mode")
    strict_thresholds: StrictThresholds = Field(default_factory=StrictThresholds)
    exploratory_thresholds: ExploratoryThresholds = Field(default_factory=ExploratoryThresholds)
    weights: Dict[str, float] = Field(
        default={"stl": 0.3, "acf": 0.3, "fourier": 0.4},
        description="Weights for composite score calculation",
    )
    confidence_labels: List[str] = Field(
        default=["High", "Medium", "Low", "None"],
        description="Confidence level labels",
    )


class AzureOpenAIConfig(BaseModel):
    """Azure OpenAI configuration."""

    api_key_env: str = Field(default="AZURE_OPENAI_API_KEY", description="Environment variable for API key")
    api_version_env: str = Field(default="AZURE_OPENAI_API_VERSION", description="Environment variable for API version")
    endpoint_env: str = Field(default="AZURE_OPENAI_ENDPOINT", description="Environment variable for endpoint")
    deployment_name: str = Field(default="gpt-4", description="Deployment name")


class OpenAIConfig(BaseModel):
    """OpenAI configuration."""

    api_key_env: str = Field(default="OPENAI_API_KEY", description="Environment variable for API key")
    model: str = Field(default="gpt-4", description="Model name")


class DebugConfig(BaseModel):
    """Configuration for debug and LLM features."""

    enable_llm: bool = Field(default=False, description="Enable LLM-based diagnostics")
    provider: Literal["azure", "openai", "auto"] = Field(default="auto", description="LLM provider")
    azure_openai: AzureOpenAIConfig = Field(default_factory=AzureOpenAIConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    anonymize_sensors: bool = Field(default=True, description="Anonymize sensor names in debug bundles")
    mask_paths: bool = Field(default=True, description="Mask file paths in debug bundles")


class OutputConfig(BaseModel):
    """Configuration for output files and directories."""

    base_dir: Path = Field(default=Path("outputs"), description="Base output directory")
    results_dir: str = Field(default="results", description="Subdirectory for result files")
    figures_dir: str = Field(default="figures", description="Subdirectory for figure files")
    logs_dir: str = Field(default="logs", description="Subdirectory for log files")
    debug_bundles_dir: str = Field(default="debug_bundles", description="Subdirectory for debug bundles")
    save_intermediate: bool = Field(default=False, description="Save intermediate processing results")
    figure_format: Literal["png", "svg", "pdf"] = Field(default="png", description="Output format for figures")
    figure_dpi: int = Field(default=150, description="DPI for figure output")

    @field_validator("base_dir", mode="before")
    @classmethod
    def convert_to_path(cls, v: Any) -> Path:
        if isinstance(v, str):
            return Path(v)
        return v


class SeasonalityConfig(BaseModel):
    """Main configuration model for Seasonality Analyzer."""

    data: DataConfig = Field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "SeasonalityConfig":
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data or {})

    def to_yaml(self, path: Path | str) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, allow_unicode=True)

    def merge_with(self, override: Dict[str, Any]) -> "SeasonalityConfig":
        """Merge current config with override dict."""
        current = self.model_dump()
        _deep_merge(current, override)
        return SeasonalityConfig.model_validate(current)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """Deep merge override dict into base dict."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def load_config(path: Optional[Path | str] = None, overrides: Optional[Dict[str, Any]] = None) -> SeasonalityConfig:
    """Load configuration from file with optional overrides.

    Args:
        path: Path to YAML configuration file. If None, uses default config.
        overrides: Dictionary of override values.

    Returns:
        SeasonalityConfig instance.
    """
    if path is not None:
        config = SeasonalityConfig.from_yaml(path)
    else:
        config = SeasonalityConfig()

    if overrides:
        config = config.merge_with(overrides)

    return config
