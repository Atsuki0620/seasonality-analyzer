"""時系列データの欠損値補間ユーティリティ"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd
from loguru import logger


def interpolate_missing(
    series: pd.Series,
    method: Literal["linear", "spline", "polynomial", "time"] = "linear",
    limit: Optional[int] = None,
    limit_direction: Literal["forward", "backward", "both"] = "both",
    order: int = 3,
) -> pd.Series:
    """時系列の欠損値を補間

    Args:
        series: タイムスタンプインデックスを持つ入力時系列
        method: 補間方法:
            - 'linear': 線形補間
            - 'spline': スプライン補間
            - 'polynomial': 多項式補間
            - 'time': 時間重み付き補間
        limit: 連続するNaNの最大補間数
        limit_direction: 補間方向（'forward', 'backward', 'both'）
        order: スプライン/多項式補間の次数

    Returns:
        欠損値が補間された系列
    """
    if series.isna().sum() == 0:
        return series

    n_missing = series.isna().sum()
    logger.debug(f"{method}を使用して{n_missing}個の欠損値を補間しています")

    if method == "linear":
        return series.interpolate(
            method="linear",
            limit=limit,
            limit_direction=limit_direction,
        )
    elif method == "spline":
        return series.interpolate(
            method="spline",
            order=order,
            limit=limit,
            limit_direction=limit_direction,
        )
    elif method == "polynomial":
        return series.interpolate(
            method="polynomial",
            order=order,
            limit=limit,
            limit_direction=limit_direction,
        )
    elif method == "time":
        return series.interpolate(
            method="time",
            limit=limit,
            limit_direction=limit_direction,
        )
    else:
        raise ValueError(f"不明な補間方法: {method}")


def forward_fill(
    series: pd.Series,
    limit: Optional[int] = None,
) -> pd.Series:
    """欠損値を前方補完

    Args:
        series: 入力時系列
        limit: 連続するNaNの最大補完数

    Returns:
        前方補完された系列
    """
    return series.ffill(limit=limit)


def backward_fill(
    series: pd.Series,
    limit: Optional[int] = None,
) -> pd.Series:
    """欠損値を後方補完

    Args:
        series: 入力時系列
        limit: 連続するNaNの最大補完数

    Returns:
        後方補完された系列
    """
    return series.bfill(limit=limit)


def fill_with_seasonal_mean(
    series: pd.Series,
    period: int = 7,
) -> pd.Series:
    """欠損値を季節平均で補完

    明確な季節性がある場合に有用（例: 曜日効果）。

    Args:
        series: タイムスタンプインデックスを持つ入力時系列
        period: 季節周期（デフォルト: 7は週次）

    Returns:
        季節的に補完された欠損値を持つ系列
    """
    result = series.copy()

    # 曜日（または周期）平均を計算
    if period == 7:
        seasonal_idx = series.index.dayofweek
    else:
        # カスタム周期の日数の剰余を使用
        day_count = (series.index - series.index[0]).days
        seasonal_idx = day_count % period

    # 各季節位置の平均を計算
    temp_df = pd.DataFrame({"value": series, "season": seasonal_idx})
    seasonal_means = temp_df.groupby("season")["value"].mean()

    # 欠損値を補完
    for i, val in enumerate(result):
        if pd.isna(val):
            season = seasonal_idx[i] if isinstance(seasonal_idx, pd.Series) else seasonal_idx
            if isinstance(season, (int, np.integer)):
                season_key = season
            else:
                season_key = season.iloc[i] if hasattr(season, "iloc") else season[i]
            if season_key in seasonal_means.index:
                result.iloc[i] = seasonal_means[season_key]

    return result


def impute_missing(
    series: pd.Series,
    method: Literal["linear", "spline", "ffill", "seasonal", "none"] = "linear",
    limit: Optional[int] = None,
    **kwargs,
) -> pd.Series:
    """指定された方法で欠損値を補間

    Args:
        series: 入力時系列
        method: 補間方法:
            - 'linear': 線形補間
            - 'spline': スプライン補間
            - 'ffill': 前方補完
            - 'seasonal': 季節平均補完
            - 'none': 補間なし
        limit: 連続するNaNの最大補完数
        **kwargs: 追加の方法固有パラメータ

    Returns:
        補間された値を持つ系列
    """
    if method == "none":
        return series

    if method == "linear":
        return interpolate_missing(series, method="linear", limit=limit)
    elif method == "spline":
        order = kwargs.get("order", 3)
        return interpolate_missing(series, method="spline", limit=limit, order=order)
    elif method == "ffill":
        return forward_fill(series, limit=limit)
    elif method == "seasonal":
        period = kwargs.get("period", 7)
        return fill_with_seasonal_mean(series, period=period)
    else:
        raise ValueError(f"不明な補間方法: {method}")


def get_missing_report(series: pd.Series) -> dict:
    """系列の欠損値に関するレポートを生成

    Args:
        series: 入力時系列

    Returns:
        欠損値統計を含む辞書
    """
    total = len(series)
    missing = series.isna().sum()

    # 最長ギャップを検索
    is_missing = series.isna()
    gaps = []
    gap_start = None

    for i, is_na in enumerate(is_missing):
        if is_na and gap_start is None:
            gap_start = i
        elif not is_na and gap_start is not None:
            gaps.append(i - gap_start)
            gap_start = None

    if gap_start is not None:
        gaps.append(len(series) - gap_start)

    return {
        "total_count": total,
        "missing_count": int(missing),
        "missing_rate": missing / total if total > 0 else 0,
        "num_gaps": len(gaps),
        "max_gap": max(gaps) if gaps else 0,
        "mean_gap": sum(gaps) / len(gaps) if gaps else 0,
    }
