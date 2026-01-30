"""時系列データのリサンプリングユーティリティ"""

from __future__ import annotations

from typing import Literal, Optional

import pandas as pd
from loguru import logger


def resample_to_daily(
    series: pd.Series,
    method: Literal["mean", "median", "ffill"] = "mean",
) -> pd.Series:
    """時系列を日次頻度にリサンプリング

    Args:
        series: タイムスタンプインデックスを持つ入力時系列
        method: 集約方法（'mean', 'median', 'ffill'）

    Returns:
        日次頻度にリサンプリングされた系列
    """
    if method == "mean":
        return series.resample("D").mean()
    elif method == "median":
        return series.resample("D").median()
    elif method == "ffill":
        return series.resample("D").ffill()
    else:
        raise ValueError(f"不明なリサンプリング方法: {method}")


def resample_and_interpolate(
    series: pd.Series,
    resample_freq: str = "D",
    resample_method: Literal["mean", "median", "ffill"] = "mean",
    interp_method: Optional[Literal["linear", "spline", "ffill"]] = "linear",
    spline_order: int = 3,
) -> pd.Series:
    """時系列をリサンプリングし、欠損値を補間

    Args:
        series: タイムスタンプインデックスを持つ入力時系列
        resample_freq: リサンプリング頻度（例: 'D'は日次）
        resample_method: リサンプリングの集約方法
        interp_method: 欠損値の補間方法
            Noneの場合は補間なし（NaNを保持）
        spline_order: スプライン補間の次数

    Returns:
        リサンプリングおよび補間された系列
    """
    # リサンプリング
    if resample_method == "mean":
        resampled = series.resample(resample_freq).mean()
    elif resample_method == "median":
        resampled = series.resample(resample_freq).median()
    elif resample_method == "ffill":
        resampled = series.resample(resample_freq).ffill()
    else:
        raise ValueError(f"不明なリサンプリング方法: {resample_method}")

    # 補間
    if interp_method is None:
        return resampled
    elif interp_method == "linear":
        return resampled.interpolate(method="linear")
    elif interp_method == "spline":
        return resampled.interpolate(method="spline", order=spline_order)
    elif interp_method == "ffill":
        return resampled.ffill()
    else:
        raise ValueError(f"不明な補間方法: {interp_method}")


def resample_dataframe(
    df: pd.DataFrame,
    sensor_cols: list[str],
    resample_freq: str = "D",
    resample_method: Literal["mean", "median", "ffill"] = "mean",
    interp_method: Optional[Literal["linear", "spline", "ffill"]] = "linear",
) -> pd.DataFrame:
    """複数のセンサー列を持つDataFrame全体をリサンプリング

    Args:
        df: タイムスタンプインデックスを持つ入力DataFrame
        sensor_cols: センサー列名のリスト
        resample_freq: リサンプリング頻度
        resample_method: 集約方法
        interp_method: 補間方法

    Returns:
        リサンプリングされたDataFrame
    """
    result = pd.DataFrame(index=df.resample(resample_freq).mean().index)

    for col in sensor_cols:
        if col in df.columns:
            result[col] = resample_and_interpolate(
                df[col],
                resample_freq=resample_freq,
                resample_method=resample_method,
                interp_method=interp_method,
            )
        else:
            logger.warning(f"列 {col} がDataFrameに見つかりません")

    return result
