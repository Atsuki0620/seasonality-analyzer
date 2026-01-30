"""PydanticベースのSeasonality Analyzer設定モデル"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    """データ読み込みの設定"""

    date_column: str = Field(default="date", description="日付列名")
    time_column: Optional[str] = Field(default="time", description="時刻列名（オプション）")
    sensor_columns: Optional[List[str]] = Field(
        default=None,
        description="センサー列名のリスト。Noneの場合は'sensor_'で始まる列を自動検出",
    )
    sensor_prefix: str = Field(default="sensor_", description="センサー列自動検出用のプレフィックス")


class PreprocessingConfig(BaseModel):
    """前処理パイプラインの設定"""

    resample_freq: str = Field(default="D", description="リサンプリング頻度（D=日次、W=週次など）")
    resample_method: Literal["mean", "median", "ffill"] = Field(
        default="mean", description="リサンプリングの集約方法"
    )
    outlier_method: Literal["iqr", "zscore", "iforest", "none"] = Field(
        default="iqr", description="外れ値検出方法"
    )
    outlier_threshold: float = Field(default=1.5, description="外れ値検出の閾値（IQRのk値、zscoreのz値）")
    impute_method: Literal["linear", "spline", "ffill", "none"] = Field(
        default="linear", description="欠損値の補間方法"
    )
    changepoint_method: Literal["pelt", "window", "none"] = Field(
        default="none", description="変化点検出方法"
    )
    changepoint_penalty: float = Field(default=10.0, description="変化点検出のペナルティパラメータ")
    min_segment_size: int = Field(default=30, description="変化点検出の最小セグメントサイズ")


class DetectionConfig(BaseModel):
    """季節性検出の設定"""

    periods: List[int] = Field(default=[7, 30, 365], description="検出対象の周期（日数）")
    methods: List[Literal["stl", "acf", "periodogram", "fourier"]] = Field(
        default=["stl", "acf", "periodogram", "fourier"],
        description="使用する検出手法",
    )
    stl_seasonal: int = Field(default=7, description="STL分解の季節性スムーザー")
    stl_robust: bool = Field(default=True, description="STLでロバスト推定を使用")
    acf_max_lag: int = Field(default=400, description="ACF計算の最大ラグ")
    acf_tolerance: int = Field(default=3, description="目標周期周辺のピーク検出許容範囲")
    periodogram_min_period: int = Field(default=2, description="Lomb-Scargle法の最小周期")
    periodogram_max_period: int = Field(default=500, description="Lomb-Scargle法の最大周期")
    fap_threshold: float = Field(default=0.05, description="有意性判定の誤警報確率閾値")


class StrictThresholds(BaseModel):
    """厳密モードスコアリングの閾値"""

    stl_strength: float = Field(default=0.5, description="STL季節強度の最小値")
    acf_peak: float = Field(default=0.3, description="ACFピーク値の最小値")
    delta_r2: float = Field(default=0.1, description="フーリエ変換のデルタR²最小値")


class ExploratoryThresholds(BaseModel):
    """探索モードスコアリングの閾値"""

    stl_strength: float = Field(default=0.2, description="STL季節強度の最小値")
    acf_peak: float = Field(default=0.15, description="ACFピーク値の最小値")
    delta_r2: float = Field(default=0.05, description="フーリエ変換のデルタR²最小値")


class ScoringConfig(BaseModel):
    """スコアリングと分類の設定"""

    mode: Literal["strict", "exploratory"] = Field(default="strict", description="スコアリングモード")
    strict_thresholds: StrictThresholds = Field(default_factory=StrictThresholds)
    exploratory_thresholds: ExploratoryThresholds = Field(default_factory=ExploratoryThresholds)
    weights: Dict[str, float] = Field(
        default={"stl": 0.3, "acf": 0.3, "fourier": 0.4},
        description="複合スコア計算の重み",
    )
    confidence_labels: List[str] = Field(
        default=["High", "Medium", "Low", "None"],
        description="信頼度レベルラベル",
    )


class AzureOpenAIConfig(BaseModel):
    """Azure OpenAIの設定"""

    api_key_env: str = Field(default="AZURE_OPENAI_API_KEY", description="APIキーの環境変数名")
    api_version_env: str = Field(default="AZURE_OPENAI_API_VERSION", description="APIバージョンの環境変数名")
    endpoint_env: str = Field(default="AZURE_OPENAI_ENDPOINT", description="エンドポイントの環境変数名")
    deployment_name: str = Field(default="gpt-4", description="デプロイメント名")


class OpenAIConfig(BaseModel):
    """OpenAIの設定"""

    api_key_env: str = Field(default="OPENAI_API_KEY", description="APIキーの環境変数名")
    model: str = Field(default="gpt-4", description="モデル名")


class DebugConfig(BaseModel):
    """デバッグとLLM機能の設定"""

    enable_llm: bool = Field(default=False, description="LLMベース診断を有効化")
    provider: Literal["azure", "openai", "auto"] = Field(default="auto", description="LLMプロバイダー")
    azure_openai: AzureOpenAIConfig = Field(default_factory=AzureOpenAIConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    anonymize_sensors: bool = Field(default=True, description="デバッグバンドル内のセンサー名を匿名化")
    mask_paths: bool = Field(default=True, description="デバッグバンドル内のファイルパスをマスク")


class OutputConfig(BaseModel):
    """出力ファイルとディレクトリの設定"""

    base_dir: Path = Field(default=Path("outputs"), description="ベース出力ディレクトリ")
    results_dir: str = Field(default="results", description="結果ファイルのサブディレクトリ")
    figures_dir: str = Field(default="figures", description="図ファイルのサブディレクトリ")
    logs_dir: str = Field(default="logs", description="ログファイルのサブディレクトリ")
    debug_bundles_dir: str = Field(default="debug_bundles", description="デバッグバンドルのサブディレクトリ")
    save_intermediate: bool = Field(default=False, description="中間処理結果を保存")
    figure_format: Literal["png", "svg", "pdf"] = Field(default="png", description="図の出力形式")
    figure_dpi: int = Field(default=150, description="図の出力DPI")

    @field_validator("base_dir", mode="before")
    @classmethod
    def convert_to_path(cls, v: Any) -> Path:
        if isinstance(v, str):
            return Path(v)
        return v


class SeasonalityConfig(BaseModel):
    """Seasonality Analyzerのメイン設定モデル"""

    data: DataConfig = Field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "SeasonalityConfig":
        """YAMLファイルから設定を読み込む"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data or {})

    def to_yaml(self, path: Path | str) -> None:
        """設定をYAMLファイルに保存"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, allow_unicode=True)

    def merge_with(self, override: Dict[str, Any]) -> "SeasonalityConfig":
        """現在の設定とオーバーライド辞書をマージ"""
        current = self.model_dump()
        _deep_merge(current, override)
        return SeasonalityConfig.model_validate(current)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """オーバーライド辞書をベース辞書に深くマージ"""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def load_config(path: Optional[Path | str] = None, overrides: Optional[Dict[str, Any]] = None) -> SeasonalityConfig:
    """ファイルから設定を読み込み、オプションでオーバーライドを適用

    Args:
        path: YAML設定ファイルのパス。Noneの場合はデフォルト設定を使用
        overrides: オーバーライド値の辞書

    Returns:
        SeasonalityConfigインスタンス
    """
    if path is not None:
        config = SeasonalityConfig.from_yaml(path)
    else:
        config = SeasonalityConfig()

    if overrides:
        config = config.merge_with(overrides)

    return config
