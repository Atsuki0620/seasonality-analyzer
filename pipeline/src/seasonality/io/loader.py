"""Seasonality Analyzerのデータ読み込みユーティリティ"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from loguru import logger

from seasonality.config import DataConfig


@dataclass
class DataHealthReport:
    """読み込まれたデータのヘルスレポート"""

    total_records: int
    date_range: Tuple[str, str]
    sensor_count: int
    sensor_columns: List[str]
    missing_counts: dict[str, int]
    missing_rate: float
    observations_per_day: dict[str, float]  # min, max, mean
    warnings: List[str] = field(default_factory=list)

    def is_healthy(self) -> bool:
        """データが基本的なヘルスチェックをパスするか確認"""
        return len(self.warnings) == 0 and self.total_records > 0

    def summary(self) -> str:
        """サマリー文字列を生成"""
        lines = [
            "=== データヘルスレポート ===",
            f"総レコード数: {self.total_records:,}",
            f"日付範囲: {self.date_range[0]} ～ {self.date_range[1]}",
            f"センサー数: {self.sensor_count}",
            f"全体欠損率: {self.missing_rate:.2%}",
            f"1日あたりの観測数: 最小={self.observations_per_day['min']:.1f}, "
            f"最大={self.observations_per_day['max']:.1f}, 平均={self.observations_per_day['mean']:.1f}",
        ]
        if self.warnings:
            lines.append("\n警告:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)


def auto_detect_sensor_columns(df: pd.DataFrame, prefix: str = "sensor_") -> List[str]:
    """プレフィックスに基づいてセンサー列を自動検出

    Args:
        df: 入力DataFrame
        prefix: マッチさせる列名プレフィックス

    Returns:
        プレフィックスにマッチする列名のリスト
    """
    return [col for col in df.columns if col.startswith(prefix)]


def load_sensor_data(
    path: Path | str,
    config: Optional[DataConfig] = None,
) -> Tuple[pd.DataFrame, DataHealthReport]:
    """CSVファイルからセンサーデータを読み込む

    Args:
        path: CSVファイルのパス
        config: データ設定。Noneの場合はデフォルトを使用

    Returns:
        (タイムスタンプをインデックスとするDataFrame, DataHealthReport)のタプル

    Raises:
        FileNotFoundError: ファイルが存在しない場合
        ValueError: 必須列が欠けている場合
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"データファイルが見つかりません: {path}")

    config = config or DataConfig()

    logger.info(f"データ読み込み中: {path}")

    # CSV読み込み
    df = pd.read_csv(path)
    logger.debug(f"{len(df)}レコードを読み込みました。列: {list(df.columns)}")

    # 必須列の検証
    if config.date_column not in df.columns:
        raise ValueError(f"日付列 '{config.date_column}' がデータに見つかりません")

    # タイムスタンプ生成
    if config.time_column and config.time_column in df.columns:
        df["timestamp"] = pd.to_datetime(
            df[config.date_column] + " " + df[config.time_column]
        )
    else:
        df["timestamp"] = pd.to_datetime(df[config.date_column])

    # タイムスタンプをインデックスに設定
    df = df.set_index("timestamp").sort_index()

    # 元の日付/時刻列を削除
    cols_to_drop = [col for col in [config.date_column, config.time_column] if col in df.columns]
    df = df.drop(columns=cols_to_drop)

    # センサー列の検出
    if config.sensor_columns:
        sensor_cols = config.sensor_columns
        missing_cols = [col for col in sensor_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"センサー列が見つかりません: {missing_cols}")
    else:
        sensor_cols = auto_detect_sensor_columns(df, config.sensor_prefix)
        if not sensor_cols:
            raise ValueError(
                f"プレフィックス '{config.sensor_prefix}' を持つセンサー列が見つかりません。"
                f"利用可能な列: {list(df.columns)}"
            )

    logger.info(f"{len(sensor_cols)}個のセンサー列を検出しました")

    # ヘルスレポート生成
    health_report = _generate_health_report(df, sensor_cols)
    logger.info(health_report.summary())

    return df, health_report


def _generate_health_report(df: pd.DataFrame, sensor_cols: List[str]) -> DataHealthReport:
    """読み込まれたデータのヘルスレポートを生成"""
    warnings = []

    # 基本統計
    total_records = len(df)
    date_range = (str(df.index.min()), str(df.index.max()))
    sensor_count = len(sensor_cols)

    # 欠損値
    missing_counts = {col: int(df[col].isna().sum()) for col in sensor_cols}
    total_cells = total_records * sensor_count
    total_missing = sum(missing_counts.values())
    missing_rate = total_missing / total_cells if total_cells > 0 else 0.0

    # 1日あたりの観測数
    daily_counts = df.groupby(df.index.date).size()
    obs_per_day = {
        "min": float(daily_counts.min()) if len(daily_counts) > 0 else 0.0,
        "max": float(daily_counts.max()) if len(daily_counts) > 0 else 0.0,
        "mean": float(daily_counts.mean()) if len(daily_counts) > 0 else 0.0,
    }

    # 警告生成
    if total_records == 0:
        warnings.append("データにレコードがありません")

    if missing_rate > 0.1:
        warnings.append(f"高い欠損率: {missing_rate:.1%}")

    # すべての値が欠損しているセンサーをチェック
    all_missing_sensors = [col for col, count in missing_counts.items() if count == total_records]
    if all_missing_sensors:
        warnings.append(f"すべての値が欠損しているセンサー: {all_missing_sensors}")

    # 日付範囲のチェック
    if total_records > 0:
        date_span = (df.index.max() - df.index.min()).days
        if date_span < 30:
            warnings.append(f"短い日付範囲: {date_span}日（推奨: >30日）")
        if date_span < 365 and any(p >= 365 for p in [7, 30, 365]):  # 一般的な周期
            warnings.append(f"日付範囲（{date_span}日）は年次季節性検出には不十分な可能性があります")

    return DataHealthReport(
        total_records=total_records,
        date_range=date_range,
        sensor_count=sensor_count,
        sensor_columns=sensor_cols,
        missing_counts=missing_counts,
        missing_rate=missing_rate,
        observations_per_day=obs_per_day,
        warnings=warnings,
    )


def validate_data_for_analysis(
    df: pd.DataFrame,
    sensor_cols: List[str],
    min_records: int = 100,
) -> Tuple[List[str], List[str]]:
    """分析に十分なデータを持つセンサーを検証

    Args:
        df: 入力DataFrame
        sensor_cols: センサー列のリスト
        min_records: 必要な非null最小レコード数

    Returns:
        (有効なセンサー, スキップされたセンサー)のタプル
    """
    valid_sensors = []
    skipped_sensors = []

    for col in sensor_cols:
        non_null_count = df[col].notna().sum()
        if non_null_count >= min_records:
            valid_sensors.append(col)
        else:
            skipped_sensors.append(col)
            logger.warning(f"{col}をスキップ: データ不足（{non_null_count} < {min_records}）")

    return valid_sensors, skipped_sensors
