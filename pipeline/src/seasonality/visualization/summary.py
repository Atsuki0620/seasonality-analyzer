"""Summary visualizations for seasonality analysis."""

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
    """Create thumbnail grid of all sensors.

    Args:
        df: DataFrame with sensor data.
        sensor_cols: List of sensor column names.
        scores_df: Optional DataFrame with scores for annotation.
        n_cols: Number of columns in grid.
        figsize: Figure size.
        save_path: Path to save figure.
        dpi: Figure resolution.

    Returns:
        Matplotlib Figure object.
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

            # Add score annotation if available
            if scores_df is not None and sensor in scores_df.index:
                score = scores_df.loc[sensor, "composite_score"] if "composite_score" in scores_df.columns else 0
                conf = scores_df.loc[sensor, "confidence_strict"] if "confidence_strict" in scores_df.columns else "N/A"
                ax.set_title(f"{sensor}\n({conf}, {score:.3f})", fontsize=8)
            else:
                ax.set_title(sensor, fontsize=8)
        else:
            ax.set_title(f"{sensor}\n(not found)", fontsize=8)
            ax.text(0.5, 0.5, "No data", ha="center", va="center")

        ax.tick_params(axis="both", labelsize=6)
        ax.grid(True, alpha=0.2)

    # Hide unused subplots
    for i in range(n_sensors, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved thumbnail grid: {save_path}")

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
    """Create detailed report figure for a single sensor.

    Args:
        df: DataFrame with sensor data.
        sensor: Sensor column name.
        periods: Target periods for analysis.
        scores: Optional dictionary with scores.
        figsize: Figure size.
        save_path: Path to save figure.
        dpi: Figure resolution.

    Returns:
        Matplotlib Figure object.
    """
    from seasonality.detection.stl import STLDetector
    from seasonality.detection.acf import ACFDetector
    from seasonality.detection.periodogram import LombScargleDetector
    from seasonality.detection.fourier import FourierDetector
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    fig = plt.figure(figsize=figsize)

    # Prepare data
    if sensor not in df.columns:
        logger.warning(f"Sensor {sensor} not found in DataFrame")
        return fig

    data = resample_and_interpolate(df[sensor], resample_freq="D", resample_method="mean", interp_method="linear")
    data = data.dropna()

    if len(data) < 100:
        logger.warning(f"Insufficient data for {sensor}: {len(data)} points")
        return fig

    # 1. Original time series
    ax1 = fig.add_subplot(4, 3, 1)
    ax1.plot(data.index, data.values, "b-", linewidth=0.5)
    ax1.set_title("Original Time Series")
    ax1.grid(True, alpha=0.3)

    # 2-4. STL decomposition
    stl_detector = STLDetector()
    period = periods[1] if len(periods) > 1 and periods[1] <= len(data) // 2 else periods[0]
    decomp = stl_detector.decompose(data, period)

    if decomp:
        ax2 = fig.add_subplot(4, 3, 2)
        ax2.plot(decomp["trend"].index, decomp["trend"].values, "g-", linewidth=1)
        ax2.set_title("Trend")
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(4, 3, 3)
        ax3.plot(decomp["seasonal"].index, decomp["seasonal"].values, "r-", linewidth=0.5)
        result = stl_detector.detect(data, [period])
        score = result.get_score(period)
        ax3.set_title(f"Seasonal (score={score:.3f})")
        ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(4, 3, 4)
        ax4.plot(decomp["residual"].index, decomp["residual"].values, "gray", linewidth=0.5)
        ax4.set_title("Residual")
        ax4.grid(True, alpha=0.3)

    # 5-6. ACF/PACF
    max_lag = min(200, len(data) // 2 - 1)
    if max_lag > 10:
        ax5 = fig.add_subplot(4, 3, 5)
        plot_acf(data, ax=ax5, lags=max_lag, alpha=0.05)
        for p in periods:
            if p <= max_lag:
                ax5.axvline(x=p, color="red", linestyle="--", alpha=0.5)
        ax5.set_title("ACF")

        ax6 = fig.add_subplot(4, 3, 6)
        plot_pacf(data, ax=ax6, lags=min(50, max_lag), alpha=0.05, method="ywm")
        ax6.set_title("PACF")

    # 7. Periodogram
    ls_detector = LombScargleDetector()
    periodogram = ls_detector.compute_full_periodogram(data)

    if periodogram:
        ax7 = fig.add_subplot(4, 3, 7)
        p_arr = np.array(periodogram["periods"])
        mask = p_arr <= 100
        ax7.plot(p_arr[mask], np.array(periodogram["powers"])[mask], "b-", linewidth=0.5)
        ax7.set_title("Periodogram (0-100d)")
        ax7.set_xlabel("Period (days)")
        for p in periods:
            if p <= 100:
                ax7.axvline(x=p, color="red", linestyle="--", alpha=0.5)
        ax7.grid(True, alpha=0.3)

    # 8. Fourier regression
    fourier_detector = FourierDetector()
    predictions = fourier_detector.fit_and_predict(data, periods)

    if predictions is not None:
        delta_r2 = fourier_detector.get_delta_r2(data, periods)
        ax8 = fig.add_subplot(4, 3, 8)
        ax8.plot(data.index, data.values, "b-", linewidth=0.3, alpha=0.5, label="Observed")
        ax8.plot(predictions.index, predictions.values, "r-", linewidth=1, label=f"Fit (ΔR²={delta_r2.get('full', 0):.4f})")
        ax8.set_title("Fourier Regression")
        ax8.legend(fontsize=8)
        ax8.grid(True, alpha=0.3)

    # 9. Scores summary text
    ax9 = fig.add_subplot(4, 3, 9)
    ax9.axis("off")

    if scores:
        summary_lines = [
            "Seasonality Scores Summary",
            "",
            f"Composite Score: {scores.get('composite_score', 'N/A'):.4f}" if isinstance(scores.get('composite_score'), float) else f"Composite Score: {scores.get('composite_score', 'N/A')}",
            f"Confidence (Strict): {scores.get('confidence_strict', 'N/A')}",
            f"Confidence (Exploratory): {scores.get('confidence_exploratory', 'N/A')}",
            "",
            f"STL Max: {scores.get('stl_max', 'N/A'):.4f}" if isinstance(scores.get('stl_max'), float) else f"STL Max: {scores.get('stl_max', 'N/A')}",
            f"ACF Max: {scores.get('acf_max', 'N/A'):.4f}" if isinstance(scores.get('acf_max'), float) else f"ACF Max: {scores.get('acf_max', 'N/A')}",
            f"Fourier Max: {scores.get('fourier_max', 'N/A'):.4f}" if isinstance(scores.get('fourier_max'), float) else f"Fourier Max: {scores.get('fourier_max', 'N/A')}",
        ]
        ax9.text(0.1, 0.5, "\n".join(summary_lines), fontsize=10, family="monospace", verticalalignment="center")
    else:
        ax9.text(0.5, 0.5, "No scores available", ha="center", va="center")

    fig.suptitle(f"{sensor} - Detailed Seasonality Report", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved detailed report: {save_path}")

    return fig


def plot_confidence_distribution(
    scores_df: pd.DataFrame,
    mode: str = "strict",
    figsize: tuple = (10, 6),
    save_path: Optional[Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    """Plot distribution of confidence levels.

    Args:
        scores_df: DataFrame with scores.
        mode: 'strict' or 'exploratory'.
        figsize: Figure size.
        save_path: Path to save figure.
        dpi: Figure resolution.

    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    confidence_col = f"confidence_{mode}"

    if confidence_col not in scores_df.columns:
        logger.warning(f"Column {confidence_col} not found")
        return fig

    # Bar chart
    counts = scores_df[confidence_col].value_counts()
    colors = {"High": "#4CAF50", "Medium": "#FF9800", "Low": "#2196F3", "None": "#9E9E9E"}
    bar_colors = [colors.get(level, "gray") for level in counts.index]

    axes[0].bar(counts.index, counts.values, color=bar_colors, edgecolor="black")
    axes[0].set_title(f"Confidence Distribution ({mode.capitalize()} Mode)")
    axes[0].set_xlabel("Confidence Level")
    axes[0].set_ylabel("Count")

    # Add count labels
    for i, (level, count) in enumerate(counts.items()):
        axes[0].text(i, count + 0.5, str(count), ha="center", fontsize=10)

    # Pie chart
    axes[1].pie(counts.values, labels=counts.index, colors=bar_colors, autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Proportion")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved confidence distribution: {save_path}")

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
    """Plot heatmap of scores by method and period.

    Args:
        scores_df: DataFrame with scores.
        methods: Methods to include.
        periods: Periods to include.
        top_n: Number of top sensors to show.
        figsize: Figure size.
        save_path: Path to save figure.
        dpi: Figure resolution.

    Returns:
        Matplotlib Figure object.
    """
    import seaborn as sns

    # Select top sensors
    if "composite_score" in scores_df.columns:
        top_sensors = scores_df.nlargest(top_n, "composite_score").index
    else:
        top_sensors = scores_df.index[:top_n]

    # Build heatmap data
    columns = []
    for method in methods:
        for period in periods:
            col = f"{method}_{period}"
            if col in scores_df.columns:
                columns.append(col)

    if not columns:
        logger.warning("No score columns found for heatmap")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        return fig

    heatmap_data = scores_df.loc[top_sensors, columns]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax)
    ax.set_title(f"Seasonality Scores Heatmap (Top {top_n} Sensors)")
    ax.set_ylabel("Sensor")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved score heatmap: {save_path}")

    return fig
