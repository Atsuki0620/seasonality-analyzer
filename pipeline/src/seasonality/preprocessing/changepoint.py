"""時系列データの変化点検出ユーティリティ"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import ruptures as rpt
from loguru import logger


@dataclass
class Segment:
    """変化点間の時系列データのセグメント"""

    start_idx: pd.Timestamp
    end_idx: pd.Timestamp
    segment_number: int
    length: int
    data: pd.Series


def detect_changepoints_pelt(
    series: pd.Series,
    model: Literal["rbf", "l1", "l2", "normal"] = "rbf",
    min_size: int = 30,
    penalty: float = 10.0,
) -> List[pd.Timestamp]:
    """PELTアルゴリズムを使用して変化点を検出

    PELT（Pruned Exact Linear Time）は線形時間計算量で
    複数の変化点を検出する厳密なアルゴリズム。

    Args:
        series: 入力時系列（NaNを含まないこと）
        model: コストモデル（'rbf', 'l1', 'l2', 'normal'）
        min_size: 最小セグメントサイズ
        penalty: 変化点数を制御するペナルティパラメータ

    Returns:
        変化点のタイムスタンプリスト
    """
    data = series.dropna()
    if len(data) < 2 * min_size:
        logger.warning(f"変化点検出にはデータが短すぎます: {len(data)} < {2 * min_size}")
        return []

    values = data.values

    algo = rpt.Pelt(model=model, min_size=min_size).fit(values)
    result = algo.predict(pen=penalty)

    # インデックスをタイムスタンプに変換（最後の要素len(data)は除外）
    change_indices = [data.index[i - 1] for i in result[:-1] if i < len(data)]

    logger.debug(f"PELTで{len(change_indices)}個の変化点を検出しました")
    return change_indices


def detect_changepoints_window(
    series: pd.Series,
    model: Literal["rbf", "l1", "l2", "normal"] = "rbf",
    min_size: int = 30,
    width: int = 50,
    penalty: float = 10.0,
) -> List[pd.Timestamp]:
    """スライディングウィンドウアルゴリズムを使用して変化点を検出

    ウィンドウベースの検出は局所的な変化により適している。

    Args:
        series: 入力時系列（NaNを含まないこと）
        model: コストモデル
        min_size: 最小セグメントサイズ
        width: ウィンドウ幅
        penalty: ペナルティパラメータ

    Returns:
        変化点のタイムスタンプリスト
    """
    data = series.dropna()
    if len(data) < 2 * min_size:
        logger.warning(f"変化点検出にはデータが短すぎます: {len(data)} < {2 * min_size}")
        return []

    values = data.values

    algo = rpt.Window(model=model, min_size=min_size, width=width).fit(values)
    result = algo.predict(pen=penalty)

    # インデックスをタイムスタンプに変換
    change_indices = [data.index[i - 1] for i in result[:-1] if i < len(data)]

    logger.debug(f"Windowで{len(change_indices)}個の変化点を検出しました")
    return change_indices


def detect_changepoints(
    series: pd.Series,
    method: Literal["pelt", "window"] = "pelt",
    model: Literal["rbf", "l1", "l2", "normal"] = "rbf",
    min_size: int = 30,
    penalty: float = 10.0,
    width: int = 50,
) -> List[pd.Timestamp]:
    """指定された方法で変化点を検出

    Args:
        series: 入力時系列
        method: 検出方法（'pelt'または'window'）
        model: rupturesのコストモデル
        min_size: 最小セグメントサイズ
        penalty: ペナルティパラメータ
        width: ウィンドウ幅（windowメソッド用）

    Returns:
        変化点のタイムスタンプリスト
    """
    if method == "pelt":
        return detect_changepoints_pelt(
            series, model=model, min_size=min_size, penalty=penalty
        )
    elif method == "window":
        return detect_changepoints_window(
            series, model=model, min_size=min_size, width=width, penalty=penalty
        )
    else:
        raise ValueError(f"不明な変化点検出方法: {method}")


def segment_by_changepoints(
    series: pd.Series,
    changepoints: List[pd.Timestamp],
) -> List[Segment]:
    """変化点で時系列をセグメントに分割

    Args:
        series: 入力時系列
        changepoints: 変化点のタイムスタンプリスト

    Returns:
        Segmentオブジェクトのリスト
    """
    segments = []
    boundaries = [series.index[0]] + sorted(changepoints) + [series.index[-1]]

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]

        # セグメントデータを取得（両端を含む）
        mask = (series.index >= start) & (series.index <= end)
        segment_data = series[mask]

        segments.append(Segment(
            start_idx=start,
            end_idx=end,
            segment_number=i + 1,
            length=len(segment_data),
            data=segment_data,
        ))

    return segments


def segment_statistics(
    series: pd.Series,
    changepoints: List[pd.Timestamp],
) -> pd.DataFrame:
    """各セグメントの統計を計算

    Args:
        series: 入力時系列
        changepoints: 変化点のタイムスタンプリスト

    Returns:
        セグメント統計を含むDataFrame
    """
    segments = segment_by_changepoints(series, changepoints)

    stats = []
    for seg in segments:
        stats.append({
            "segment": seg.segment_number,
            "start": seg.start_idx,
            "end": seg.end_idx,
            "n_points": seg.length,
            "mean": seg.data.mean(),
            "std": seg.data.std(),
            "min": seg.data.min(),
            "max": seg.data.max(),
            "median": seg.data.median(),
        })

    return pd.DataFrame(stats)
