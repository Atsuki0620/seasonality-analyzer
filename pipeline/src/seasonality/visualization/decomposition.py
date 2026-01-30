"""季節性分析のための分解可視化モジュール。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from loguru import logger

from seasonality.detection.stl import STLDetector
from seasonality.detection.acf import ACFDetector
from seasonality.detection.periodogram import LombScargleDetector
from seasonality.detection.fourier import FourierDetector


def plot_stl_decomposition(
    series: pd.Series,
    period: int,
    title: Optional[str] = None,
    figsize: tuple = (14, 10),
    save_path: Optional[Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    """STL分解成分をプロットします。

    Args:
        series: 入力時系列データ。
        period: 季節周期。
        title: プロットのタイトル。
        figsize: 図のサイズ。
        save_path: 図を保存するパス。
        dpi: 図の解像度。

    Returns:
        Matplotlib Figureオブジェクト。
    """
    detector = STLDetector()
    decomposition = detector.decompose(series, period)

    if decomposition is None:
        logger.warning("STL分解に失敗しました。空の図を返します")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "STL分解に失敗しました", ha="center", va="center")
        return fig

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    components = [
        ("観測値", decomposition["observed"], "blue"),
        ("トレンド", decomposition["trend"], "green"),
        ("季節成分", decomposition["seasonal"], "red"),
        ("残差", decomposition["residual"], "gray"),
    ]

    for ax, (name, data, color) in zip(axes, components):
        ax.plot(data.index, data.values, color=color, linewidth=0.5)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    plt.xticks(rotation=45)

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    else:
        fig.suptitle(f"STL分解 (周期={period})", fontsize=12, y=1.02)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"STLプロットを保存しました: {save_path}")

    return fig


def plot_acf_pacf(
    series: pd.Series,
    max_lag: int = 200,
    target_periods: Optional[List[int]] = None,
    title: Optional[str] = None,
    figsize: tuple = (14, 6),
    save_path: Optional[Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    """ACFとPACFを周期マーカー付きでプロットします。

    Args:
        series: 入力時系列データ。
        max_lag: ACFの最大ラグ。
        target_periods: プロット上にマークする周期。
        title: プロットのタイトル。
        figsize: 図のサイズ。
        save_path: 図を保存するパス。
        dpi: 図の解像度。

    Returns:
        Matplotlib Figureオブジェクト。
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    target_periods = target_periods or [7, 30, 365]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    clean_series = series.dropna()
    n = len(clean_series)
    max_lag = min(max_lag, n // 2 - 1)

    if max_lag < 10:
        logger.warning("系列が短すぎて有意なACFプロットを作成できません")
        axes[0].text(0.5, 0.5, "データ不足", ha="center", va="center")
        axes[1].text(0.5, 0.5, "データ不足", ha="center", va="center")
        return fig

    # ACFプロット
    plot_acf(clean_series, ax=axes[0], lags=max_lag, alpha=0.05)
    axes[0].set_title("自己相関(ACF)")

    # 周期マーカーを追加
    for period in target_periods:
        if period <= max_lag:
            axes[0].axvline(x=period, color="red", linestyle="--", alpha=0.5, label=f"{period}d")
    axes[0].legend(loc="upper right", fontsize=8)

    # PACFプロット
    pacf_lag = min(50, max_lag)
    plot_pacf(clean_series, ax=axes[1], lags=pacf_lag, alpha=0.05, method="ywm")
    axes[1].set_title("偏自己相関(PACF)")

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"ACF/PACFプロットを保存しました: {save_path}")

    return fig


def plot_periodogram(
    series: pd.Series,
    target_periods: Optional[List[int]] = None,
    max_period_display: int = 400,
    title: Optional[str] = None,
    figsize: tuple = (14, 5),
    save_path: Optional[Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    """Lomb-Scargle周期図をプロットします。

    Args:
        series: 入力時系列データ。
        target_periods: プロット上にマークする周期。
        max_period_display: x軸に表示する最大周期。
        title: プロットのタイトル。
        figsize: 図のサイズ。
        save_path: 図を保存するパス。
        dpi: 図の解像度。

    Returns:
        Matplotlib Figureオブジェクト。
    """
    detector = LombScargleDetector()
    periodogram = detector.compute_full_periodogram(series)

    fig, ax = plt.subplots(figsize=figsize)

    if periodogram is None:
        ax.text(0.5, 0.5, "周期図の計算に失敗しました", ha="center", va="center")
        return fig

    periods = np.array(periodogram["periods"])
    powers = np.array(periodogram["powers"])

    # 表示範囲でフィルタリング
    mask = periods <= max_period_display
    ax.plot(periods[mask], powers[mask], "b-", linewidth=0.5)

    # ターゲット周期をマーク
    target_periods = target_periods or [7, 30, 365]
    for period in target_periods:
        if period <= max_period_display:
            ax.axvline(x=period, color="red", linestyle="--", alpha=0.5)
            ax.annotate(f"{period}d", (period, ax.get_ylim()[1] * 0.9), fontsize=9, color="red")

    ax.set_xlabel("周期(日)")
    ax.set_ylabel("パワー")
    ax.set_title(title or "Lomb-Scargle周期図")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"周期図を保存しました: {save_path}")

    return fig


def plot_fourier_fit(
    series: pd.Series,
    periods: List[int],
    title: Optional[str] = None,
    figsize: tuple = (14, 5),
    save_path: Optional[Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    """フーリエ回帰フィットをプロットします。

    Args:
        series: 入力時系列データ。
        periods: フーリエモデルに含める周期。
        title: プロットのタイトル。
        figsize: 図のサイズ。
        save_path: 図を保存するパス。
        dpi: 図の解像度。

    Returns:
        Matplotlib Figureオブジェクト。
    """
    detector = FourierDetector()
    predictions = detector.fit_and_predict(series, periods)

    fig, ax = plt.subplots(figsize=figsize)

    clean_series = series.dropna()

    # 元データをプロット
    ax.plot(clean_series.index, clean_series.values, "b-", linewidth=0.3, alpha=0.5, label="観測値")

    # フーリエフィットをプロット
    if predictions is not None:
        delta_r2 = detector.get_delta_r2(series, periods)
        full_delta = delta_r2.get("full", 0)
        ax.plot(predictions.index, predictions.values, "r-", linewidth=1, label=f"フーリエフィット (ΔR²={full_delta:.4f})")

    ax.set_xlabel("日付")
    ax.set_ylabel("値")
    ax.set_title(title or f"フーリエ回帰 (周期={periods})")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"フーリエフィットを保存しました: {save_path}")

    return fig


def plot_all_methods(
    series: pd.Series,
    periods: List[int],
    sensor_name: str = "センサー",
    figsize: tuple = (18, 16),
    save_path: Optional[Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    """全ての検出手法を含む包括的なプロットを作成します。

    Args:
        series: 入力時系列データ。
        periods: ターゲット周期。
        sensor_name: タイトルに使用する名前。
        figsize: 図のサイズ。
        save_path: 図を保存するパス。
        dpi: 図の解像度。

    Returns:
        Matplotlib Figureオブジェクト。
    """
    fig = plt.figure(figsize=figsize)

    clean_series = series.dropna()

    # 1. 時系列プロット
    ax1 = fig.add_subplot(4, 2, 1)
    ax1.plot(clean_series.index, clean_series.values, "b-", linewidth=0.5)
    ax1.set_title("元の時系列")
    ax1.grid(True, alpha=0.3)

    # 2. STL分解（トレンドと季節成分）
    stl_detector = STLDetector()
    period = periods[1] if len(periods) > 1 else periods[0]  # 可能であれば30日周期を使用
    decomp = stl_detector.decompose(clean_series, period)

    if decomp:
        ax2 = fig.add_subplot(4, 2, 2)
        ax2.plot(decomp["trend"].index, decomp["trend"].values, "g-", linewidth=1)
        ax2.set_title("トレンド成分")
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(4, 2, 3)
        ax3.plot(decomp["seasonal"].index, decomp["seasonal"].values, "r-", linewidth=0.5)
        ax3.set_title(f"季節成分 (周期={period}d)")
        ax3.grid(True, alpha=0.3)

    # 3. ACF
    from statsmodels.graphics.tsaplots import plot_acf
    max_lag = min(200, len(clean_series) // 2 - 1)
    if max_lag > 10:
        ax4 = fig.add_subplot(4, 2, 4)
        plot_acf(clean_series, ax=ax4, lags=max_lag, alpha=0.05)
        for p in periods:
            if p <= max_lag:
                ax4.axvline(x=p, color="red", linestyle="--", alpha=0.5)
        ax4.set_title("自己相関(ACF)")

    # 4. 周期図
    ls_detector = LombScargleDetector()
    periodogram = ls_detector.compute_full_periodogram(clean_series)

    if periodogram:
        ax5 = fig.add_subplot(4, 2, 5)
        p_arr = np.array(periodogram["periods"])
        mask = p_arr <= 100
        ax5.plot(p_arr[mask], np.array(periodogram["powers"])[mask], "b-", linewidth=0.5)
        ax5.set_title("周期図 (0-100日)")
        ax5.set_xlabel("周期(日)")
        for p in periods:
            if p <= 100:
                ax5.axvline(x=p, color="red", linestyle="--", alpha=0.5)
        ax5.grid(True, alpha=0.3)

    # 5. フーリエフィット
    fourier_detector = FourierDetector()
    predictions = fourier_detector.fit_and_predict(clean_series, periods)

    if predictions is not None:
        delta_r2 = fourier_detector.get_delta_r2(clean_series, periods)
        ax6 = fig.add_subplot(4, 2, 6)
        ax6.plot(clean_series.index, clean_series.values, "b-", linewidth=0.3, alpha=0.5, label="観測値")
        ax6.plot(predictions.index, predictions.values, "r-", linewidth=1, label=f"フーリエ (ΔR²={delta_r2.get('full', 0):.4f})")
        ax6.set_title("フーリエ回帰フィット")
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)

    # 6. 残差ヒストグラム
    if decomp:
        ax7 = fig.add_subplot(4, 2, 7)
        ax7.hist(decomp["residual"].dropna(), bins=50, edgecolor="black", alpha=0.7)
        ax7.set_title("残差分布")
        ax7.set_xlabel("残差値")

    fig.suptitle(f"{sensor_name} - 季節性分析", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"包括的プロットを保存しました: {save_path}")

    return fig
