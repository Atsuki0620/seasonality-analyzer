"""Decomposition visualizations for seasonality analysis."""

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
    """Plot STL decomposition components.

    Args:
        series: Input time series.
        period: Seasonal period.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save figure.
        dpi: Figure resolution.

    Returns:
        Matplotlib Figure object.
    """
    detector = STLDetector()
    decomposition = detector.decompose(series, period)

    if decomposition is None:
        logger.warning("STL decomposition failed, returning empty figure")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "STL decomposition failed", ha="center", va="center")
        return fig

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    components = [
        ("Observed", decomposition["observed"], "blue"),
        ("Trend", decomposition["trend"], "green"),
        ("Seasonal", decomposition["seasonal"], "red"),
        ("Residual", decomposition["residual"], "gray"),
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
        fig.suptitle(f"STL Decomposition (period={period})", fontsize=12, y=1.02)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved STL plot: {save_path}")

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
    """Plot ACF and PACF with period markers.

    Args:
        series: Input time series.
        max_lag: Maximum lag for ACF.
        target_periods: Periods to mark on plot.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save figure.
        dpi: Figure resolution.

    Returns:
        Matplotlib Figure object.
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    target_periods = target_periods or [7, 30, 365]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    clean_series = series.dropna()
    n = len(clean_series)
    max_lag = min(max_lag, n // 2 - 1)

    if max_lag < 10:
        logger.warning("Series too short for meaningful ACF plot")
        axes[0].text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        axes[1].text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        return fig

    # ACF plot
    plot_acf(clean_series, ax=axes[0], lags=max_lag, alpha=0.05)
    axes[0].set_title("Autocorrelation Function (ACF)")

    # Add period markers
    for period in target_periods:
        if period <= max_lag:
            axes[0].axvline(x=period, color="red", linestyle="--", alpha=0.5, label=f"{period}d")
    axes[0].legend(loc="upper right", fontsize=8)

    # PACF plot
    pacf_lag = min(50, max_lag)
    plot_pacf(clean_series, ax=axes[1], lags=pacf_lag, alpha=0.05, method="ywm")
    axes[1].set_title("Partial Autocorrelation Function (PACF)")

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved ACF/PACF plot: {save_path}")

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
    """Plot Lomb-Scargle periodogram.

    Args:
        series: Input time series.
        target_periods: Periods to mark on plot.
        max_period_display: Maximum period to display on x-axis.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save figure.
        dpi: Figure resolution.

    Returns:
        Matplotlib Figure object.
    """
    detector = LombScargleDetector()
    periodogram = detector.compute_full_periodogram(series)

    fig, ax = plt.subplots(figsize=figsize)

    if periodogram is None:
        ax.text(0.5, 0.5, "Periodogram computation failed", ha="center", va="center")
        return fig

    periods = np.array(periodogram["periods"])
    powers = np.array(periodogram["powers"])

    # Filter to display range
    mask = periods <= max_period_display
    ax.plot(periods[mask], powers[mask], "b-", linewidth=0.5)

    # Mark target periods
    target_periods = target_periods or [7, 30, 365]
    for period in target_periods:
        if period <= max_period_display:
            ax.axvline(x=period, color="red", linestyle="--", alpha=0.5)
            ax.annotate(f"{period}d", (period, ax.get_ylim()[1] * 0.9), fontsize=9, color="red")

    ax.set_xlabel("Period (days)")
    ax.set_ylabel("Power")
    ax.set_title(title or "Lomb-Scargle Periodogram")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved periodogram: {save_path}")

    return fig


def plot_fourier_fit(
    series: pd.Series,
    periods: List[int],
    title: Optional[str] = None,
    figsize: tuple = (14, 5),
    save_path: Optional[Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    """Plot Fourier regression fit.

    Args:
        series: Input time series.
        periods: Periods to include in Fourier model.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save figure.
        dpi: Figure resolution.

    Returns:
        Matplotlib Figure object.
    """
    detector = FourierDetector()
    predictions = detector.fit_and_predict(series, periods)

    fig, ax = plt.subplots(figsize=figsize)

    clean_series = series.dropna()

    # Plot original data
    ax.plot(clean_series.index, clean_series.values, "b-", linewidth=0.3, alpha=0.5, label="Observed")

    # Plot Fourier fit
    if predictions is not None:
        delta_r2 = detector.get_delta_r2(series, periods)
        full_delta = delta_r2.get("full", 0)
        ax.plot(predictions.index, predictions.values, "r-", linewidth=1, label=f"Fourier fit (ΔR²={full_delta:.4f})")

    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title(title or f"Fourier Regression (periods={periods})")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved Fourier fit: {save_path}")

    return fig


def plot_all_methods(
    series: pd.Series,
    periods: List[int],
    sensor_name: str = "Sensor",
    figsize: tuple = (18, 16),
    save_path: Optional[Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    """Create comprehensive plot with all detection methods.

    Args:
        series: Input time series.
        periods: Target periods.
        sensor_name: Name for title.
        figsize: Figure size.
        save_path: Path to save figure.
        dpi: Figure resolution.

    Returns:
        Matplotlib Figure object.
    """
    fig = plt.figure(figsize=figsize)

    clean_series = series.dropna()

    # 1. Time series plot
    ax1 = fig.add_subplot(4, 2, 1)
    ax1.plot(clean_series.index, clean_series.values, "b-", linewidth=0.5)
    ax1.set_title("Original Time Series")
    ax1.grid(True, alpha=0.3)

    # 2. STL Decomposition (trend and seasonal)
    stl_detector = STLDetector()
    period = periods[1] if len(periods) > 1 else periods[0]  # Use 30-day if available
    decomp = stl_detector.decompose(clean_series, period)

    if decomp:
        ax2 = fig.add_subplot(4, 2, 2)
        ax2.plot(decomp["trend"].index, decomp["trend"].values, "g-", linewidth=1)
        ax2.set_title("Trend Component")
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(4, 2, 3)
        ax3.plot(decomp["seasonal"].index, decomp["seasonal"].values, "r-", linewidth=0.5)
        ax3.set_title(f"Seasonal Component (period={period}d)")
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
        ax4.set_title("ACF")

    # 4. Periodogram
    ls_detector = LombScargleDetector()
    periodogram = ls_detector.compute_full_periodogram(clean_series)

    if periodogram:
        ax5 = fig.add_subplot(4, 2, 5)
        p_arr = np.array(periodogram["periods"])
        mask = p_arr <= 100
        ax5.plot(p_arr[mask], np.array(periodogram["powers"])[mask], "b-", linewidth=0.5)
        ax5.set_title("Periodogram (0-100 days)")
        ax5.set_xlabel("Period (days)")
        for p in periods:
            if p <= 100:
                ax5.axvline(x=p, color="red", linestyle="--", alpha=0.5)
        ax5.grid(True, alpha=0.3)

    # 5. Fourier fit
    fourier_detector = FourierDetector()
    predictions = fourier_detector.fit_and_predict(clean_series, periods)

    if predictions is not None:
        delta_r2 = fourier_detector.get_delta_r2(clean_series, periods)
        ax6 = fig.add_subplot(4, 2, 6)
        ax6.plot(clean_series.index, clean_series.values, "b-", linewidth=0.3, alpha=0.5, label="Observed")
        ax6.plot(predictions.index, predictions.values, "r-", linewidth=1, label=f"Fourier (ΔR²={delta_r2.get('full', 0):.4f})")
        ax6.set_title("Fourier Regression Fit")
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)

    # 6. Residuals histogram
    if decomp:
        ax7 = fig.add_subplot(4, 2, 7)
        ax7.hist(decomp["residual"].dropna(), bins=50, edgecolor="black", alpha=0.7)
        ax7.set_title("Residual Distribution")
        ax7.set_xlabel("Residual Value")

    fig.suptitle(f"{sensor_name} - Seasonality Analysis", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved comprehensive plot: {save_path}")

    return fig
