"""時系列データの外れ値検出ユーティリティ"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from loguru import logger


def detect_outliers_iqr(
    series: pd.Series,
    k: float = 1.5,
) -> pd.Series:
    """IQR法を使用して外れ値を検出

    Args:
        series: 入力時系列
        k: IQRの乗数（デフォルト: 1.5）

    Returns:
        外れ値を示すブール型系列
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR
    return (series < lower) | (series > upper)


def detect_outliers_zscore(
    series: pd.Series,
    threshold: float = 3.0,
) -> pd.Series:
    """Zスコア法を使用して外れ値を検出

    Args:
        series: 入力時系列
        threshold: Zスコア閾値（デフォルト: 3.0）

    Returns:
        外れ値を示すブール型系列
    """
    data = series.dropna()
    if len(data) == 0:
        return pd.Series(False, index=series.index)

    z_scores = np.abs(stats.zscore(data))
    outlier_idx = data.index[z_scores > threshold]
    return pd.Series(series.index.isin(outlier_idx), index=series.index)


def detect_outliers_iforest(
    series: pd.Series,
    contamination: float = 0.05,
    random_state: int = 42,
) -> pd.Series:
    """Isolation Forestを使用して外れ値を検出

    Args:
        series: 入力時系列
        contamination: 外れ値の予想割合
        random_state: 再現性のための乱数シード

    Returns:
        外れ値を示すブール型系列
    """
    data = series.dropna()
    if len(data) == 0:
        return pd.Series(False, index=series.index)

    X = data.values.reshape(-1, 1)
    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
    )
    predictions = iso.fit_predict(X)
    outlier_idx = data.index[predictions == -1]
    return pd.Series(series.index.isin(outlier_idx), index=series.index)


def detect_outliers(
    series: pd.Series,
    method: Literal["iqr", "zscore", "iforest"] = "iqr",
    threshold: Optional[float] = None,
    **kwargs,
) -> pd.Series:
    """指定された方法で外れ値を検出

    Args:
        series: 入力時系列
        method: 検出方法（'iqr', 'zscore', 'iforest'）
        threshold: 方法固有の閾値
        **kwargs: 追加の方法固有パラメータ

    Returns:
        外れ値を示すブール型系列
    """
    if method == "iqr":
        k = threshold if threshold is not None else 1.5
        return detect_outliers_iqr(series, k=k)
    elif method == "zscore":
        t = threshold if threshold is not None else 3.0
        return detect_outliers_zscore(series, threshold=t)
    elif method == "iforest":
        contamination = threshold if threshold is not None else 0.05
        return detect_outliers_iforest(
            series,
            contamination=contamination,
            **kwargs,
        )
    else:
        raise ValueError(f"不明な外れ値検出方法: {method}")


def remove_outliers(
    series: pd.Series,
    method: Literal["iqr", "zscore", "iforest"] = "iqr",
    threshold: Optional[float] = None,
    replace_with: Literal["nan", "median", "interpolate"] = "nan",
    **kwargs,
) -> pd.Series:
    """時系列の外れ値を削除または置換

    Args:
        series: 入力時系列
        method: 検出方法
        threshold: 検出閾値
        replace_with: 外れ値の処理方法:
            - 'nan': NaNで置換
            - 'median': 系列の中央値で置換
            - 'interpolate': 補間値で置換

    Returns:
        外れ値が削除/置換された系列
    """
    outliers = detect_outliers(series, method=method, threshold=threshold, **kwargs)
    result = series.copy()

    n_outliers = outliers.sum()
    if n_outliers > 0:
        logger.debug(f"{n_outliers}個の外れ値を検出しました（{n_outliers/len(series)*100:.2f}%）")

        if replace_with == "nan":
            result[outliers] = np.nan
        elif replace_with == "median":
            result[outliers] = series.median()
        elif replace_with == "interpolate":
            result[outliers] = np.nan
            result = result.interpolate(method="linear")
        else:
            raise ValueError(f"不明なreplace_with値: {replace_with}")

    return result


def detect_outliers_rolling(
    series: pd.Series,
    window: int = 30,
    threshold: float = 3.0,
) -> pd.Series:
    """ローリングウィンドウZスコアを使用して外れ値を検出

    非定常データの局所的な異常を検出するのに有用。

    Args:
        series: 入力時系列
        window: ローリングウィンドウサイズ
        threshold: Zスコア閾値

    Returns:
        外れ値を示すブール型系列
    """
    rolling_mean = series.rolling(window=window, center=True, min_periods=1).mean()
    rolling_std = series.rolling(window=window, center=True, min_periods=1).std()

    # ゼロ除算を回避
    rolling_std = rolling_std.replace(0, np.nan)

    z_scores = np.abs((series - rolling_mean) / rolling_std)
    return z_scores > threshold
