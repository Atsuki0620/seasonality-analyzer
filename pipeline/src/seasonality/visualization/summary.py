"""季節性分析のためのサマリー可視化モジュール。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from loguru import logger

from seasonality.preprocessing.resample import resample_and_interpolate


def create_thumbnail_grid(
    df: pd.DataFrame,
    sensor_cols: List[str],
    scores_df: Optional[pd.DataFrame] = None,
    n_cols: int = 4,
    figsize: tuple = (16, 20),
    save_path: Optional[Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    """全センサーのサムネイルグリッドを作成します。

    Args:
        df: センサーデータを含むDataFrame。
        sensor_cols: センサー列名のリスト。
        scores_df: アノテーション用のスコアを含むオプションのDataFrame。
        n_cols: グリッドの列数。
        figsize: 図のサイズ。
        save_path: 図を保存するパス。
        dpi: 図の解像度。

    Returns:
        Matplotlib Figureオブジェクト。
    """
    n_sensors = len(sensor_cols)
    n_rows = (n_sensors + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, sensor in enumerate(sensor_cols):
        ax = axes[i]

        if sensor in df.columns:
            data = resample_and_interpolate(df[sensor], resample_freq="D", resample_method="mean", interp_method="linear")
            data = data.dropna()

            ax.plot(data.index, data.values, "b-", linewidth=0.3)

            # 利用可能な場合はスコアアノテーションを追加
            if scores_df is not None and sensor in scores_df.index:
                score = scores_df.loc[sensor, "composite_score"] if "composite_score" in scores_df.columns else 0
                conf = scores_df.loc[sensor, "confidence_strict"] if "confidence_strict" in scores_df.columns else "N/A"
                ax.set_title(f"{sensor}\n({conf}, {score:.3f})", fontsize=8)
            else:
                ax.set_title(sensor, fontsize=8)
        else:
            ax.set_title(f"{sensor}\n(見つかりません)", fontsize=8)
            ax.text(0.5, 0.5, "データなし", ha="center", va="center")

        ax.tick_params(axis="both", labelsize=6)
        ax.grid(True, alpha=0.2)

    # 未使用のサブプロットを非表示
    for i in range(n_sensors, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"サムネイルグリッドを保存しました: {save_path}")

    return fig


def create_detailed_report(
    df: pd.DataFrame,
    sensor: str,
    periods: List[int],
    scores: Optional[Dict] = None,
    figsize: tuple = (18, 16),
    save_path: Optional[Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    """単一センサーの詳細レポート図を作成します。

    Args:
        df: センサーデータを含むDataFrame。
        sensor: センサー列名。
        periods: 分析のターゲット周期。
        scores: スコアを含むオプションの辞書。
        figsize: 図のサイズ。
        save_path: 図を保存するパス。
        dpi: 図の解像度。

    Returns:
        Matplotlib Figureオブジェクト。
    """
    from seasonality.detection.stl import STLDetector
    from seasonality.detection.acf import ACFDetector
    from seasonality.detection.periodogram import LombScargleDetector
    from seasonality.detection.fourier import FourierDetector
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    fig = plt.figure(figsize=figsize)

    # データ準備
    if sensor not in df.columns:
        logger.warning(f"センサー {sensor} がDataFrameに見つかりません")
        return fig

    data = resample_and_interpolate(df[sensor], resample_freq="D", resample_method="mean", interp_method="linear")
    data = data.dropna()

    if len(data) < 100:
        logger.warning(f"{sensor}のデータが不足しています: {len(data)}ポイント")
        return fig

    # 1. 元の時系列
    ax1 = fig.add_subplot(4, 3, 1)
    ax1.plot(data.index, data.values, "b-", linewidth=0.5)
    ax1.set_title("元の時系列")
    ax1.grid(True, alpha=0.3)

    # 2-4. STL分解
    stl_detector = STLDetector()
    period = periods[1] if len(periods) > 1 and periods[1] <= len(data) // 2 else periods[0]
    decomp = stl_detector.decompose(data, period)

    if decomp:
        ax2 = fig.add_subplot(4, 3, 2)
        ax2.plot(decomp["trend"].index, decomp["trend"].values, "g-", linewidth=1)
        ax2.set_title("トレンド")
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(4, 3, 3)
        ax3.plot(decomp["seasonal"].index, decomp["seasonal"].values, "r-", linewidth=0.5)
        result = stl_detector.detect(data, [period])
        score = result.get_score(period)
        ax3.set_title(f"季節成分 (スコア={score:.3f})")
        ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(4, 3, 4)
        ax4.plot(decomp["residual"].index, decomp["residual"].values, "gray", linewidth=0.5)
        ax4.set_title("残差")
        ax4.grid(True, alpha=0.3)

    # 5-6. ACF/PACF
    max_lag = min(200, len(data) // 2 - 1)
    if max_lag > 10:
        ax5 = fig.add_subplot(4, 3, 5)
        plot_acf(data, ax=ax5, lags=max_lag, alpha=0.05)
        for p in periods:
            if p <= max_lag:
                ax5.axvline(x=p, color="red", linestyle="--", alpha=0.5)
        ax5.set_title("自己相関(ACF)")

        ax6 = fig.add_subplot(4, 3, 6)
        plot_pacf(data, ax=ax6, lags=min(50, max_lag), alpha=0.05, method="ywm")
        ax6.set_title("偏自己相関(PACF)")

    # 7. 周期図
    ls_detector = LombScargleDetector()
    periodogram = ls_detector.compute_full_periodogram(data)

    if periodogram:
        ax7 = fig.add_subplot(4, 3, 7)
        p_arr = np.array(periodogram["periods"])
        mask = p_arr <= 100
        ax7.plot(p_arr[mask], np.array(periodogram["powers"])[mask], "b-", linewidth=0.5)
        ax7.set_title("周期図 (0-100d)")
        ax7.set_xlabel("周期(日)")
        for p in periods:
            if p <= 100:
                ax7.axvline(x=p, color="red", linestyle="--", alpha=0.5)
        ax7.grid(True, alpha=0.3)

    # 8. フーリエ回帰
    fourier_detector = FourierDetector()
    predictions = fourier_detector.fit_and_predict(data, periods)

    if predictions is not None:
        delta_r2 = fourier_detector.get_delta_r2(data, periods)
        ax8 = fig.add_subplot(4, 3, 8)
        ax8.plot(data.index, data.values, "b-", linewidth=0.3, alpha=0.5, label="観測値")
        ax8.plot(predictions.index, predictions.values, "r-", linewidth=1, label=f"フィット (ΔR²={delta_r2.get('full', 0):.4f})")
        ax8.set_title("フーリエ回帰")
        ax8.legend(fontsize=8)
        ax8.grid(True, alpha=0.3)

    # 9. スコアサマリーテキスト
    ax9 = fig.add_subplot(4, 3, 9)
    ax9.axis("off")

    if scores:
        summary_lines = [
            "季節性スコアサマリー",
            "",
            f"総合スコア: {scores.get('composite_score', 'N/A'):.4f}" if isinstance(scores.get('composite_score'), float) else f"総合スコア: {scores.get('composite_score', 'N/A')}",
            f"信頼度(厳格): {scores.get('confidence_strict', 'N/A')}",
            f"信頼度(探索的): {scores.get('confidence_exploratory', 'N/A')}",
            "",
            f"STL最大: {scores.get('stl_max', 'N/A'):.4f}" if isinstance(scores.get('stl_max'), float) else f"STL最大: {scores.get('stl_max', 'N/A')}",
            f"ACF最大: {scores.get('acf_max', 'N/A'):.4f}" if isinstance(scores.get('acf_max'), float) else f"ACF最大: {scores.get('acf_max', 'N/A')}",
            f"Fourier最大: {scores.get('fourier_max', 'N/A'):.4f}" if isinstance(scores.get('fourier_max'), float) else f"Fourier最大: {scores.get('fourier_max', 'N/A')}",
        ]
        ax9.text(0.1, 0.5, "\n".join(summary_lines), fontsize=10, family="monospace", verticalalignment="center")
    else:
        ax9.text(0.5, 0.5, "スコアが利用できません", ha="center", va="center")

    fig.suptitle(f"{sensor} - 詳細季節性レポート", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"詳細レポートを保存しました: {save_path}")

    return fig


def plot_confidence_distribution(
    scores_df: pd.DataFrame,
    mode: str = "strict",
    figsize: tuple = (10, 6),
    save_path: Optional[Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    """信頼度レベルの分布をプロットします。

    Args:
        scores_df: スコアを含むDataFrame。
        mode: 'strict'または'exploratory'。
        figsize: 図のサイズ。
        save_path: 図を保存するパス。
        dpi: 図の解像度。

    Returns:
        Matplotlib Figureオブジェクト。
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    confidence_col = f"confidence_{mode}"

    if confidence_col not in scores_df.columns:
        logger.warning(f"列 {confidence_col} が見つかりません")
        return fig

    # 棒グラフ
    counts = scores_df[confidence_col].value_counts()
    colors = {"High": "#4CAF50", "Medium": "#FF9800", "Low": "#2196F3", "None": "#9E9E9E"}
    bar_colors = [colors.get(level, "gray") for level in counts.index]

    axes[0].bar(counts.index, counts.values, color=bar_colors, edgecolor="black")
    axes[0].set_title(f"信頼度分布 ({mode.capitalize()}モード)")
    axes[0].set_xlabel("信頼度レベル")
    axes[0].set_ylabel("カウント")

    # カウントラベルを追加
    for i, (level, count) in enumerate(counts.items()):
        axes[0].text(i, count + 0.5, str(count), ha="center", fontsize=10)

    # 円グラフ
    axes[1].pie(counts.values, labels=counts.index, colors=bar_colors, autopct="%1.1f%%", startangle=90)
    axes[1].set_title("割合")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"信頼度分布を保存しました: {save_path}")

    return fig


def plot_score_heatmap(
    scores_df: pd.DataFrame,
    methods: List[str] = ["stl", "acf", "periodogram", "fourier"],
    periods: List[int] = [7, 30, 365],
    top_n: int = 20,
    figsize: tuple = (12, 10),
    save_path: Optional[Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    """手法と周期別のスコアヒートマップをプロットします。

    Args:
        scores_df: スコアを含むDataFrame。
        methods: 含める手法。
        periods: 含める周期。
        top_n: 表示する上位センサーの数。
        figsize: 図のサイズ。
        save_path: 図を保存するパス。
        dpi: 図の解像度。

    Returns:
        Matplotlib Figureオブジェクト。
    """
    import seaborn as sns

    # 上位センサーを選択
    if "composite_score" in scores_df.columns:
        top_sensors = scores_df.nlargest(top_n, "composite_score").index
    else:
        top_sensors = scores_df.index[:top_n]

    # ヒートマップデータを構築
    columns = []
    for method in methods:
        for period in periods:
            col = f"{method}_{period}"
            if col in scores_df.columns:
                columns.append(col)

    if not columns:
        logger.warning("ヒートマップ用のスコア列が見つかりません")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "データが利用できません", ha="center", va="center")
        return fig

    heatmap_data = scores_df.loc[top_sensors, columns]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax)
    ax.set_title(f"季節性スコアヒートマップ (上位{top_n}センサー)")
    ax.set_ylabel("センサー")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"スコアヒートマップを保存しました: {save_path}")

    return fig
