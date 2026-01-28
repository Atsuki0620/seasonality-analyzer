"""HTML report generation for seasonality analysis."""

from __future__ import annotations

import base64
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from seasonality.config import OutputConfig


def figure_to_base64(fig: plt.Figure, format: str = "png") -> str:
    """Convert matplotlib figure to base64 encoded string.

    Args:
        fig: Matplotlib figure.
        format: Image format ('png' or 'svg').

    Returns:
        Base64 encoded string.
    """
    buffer = BytesIO()
    fig.savefig(buffer, format=format, bbox_inches="tight", dpi=150)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    buffer.close()
    return image_base64


def create_report_figures(
    df: pd.DataFrame,
    scores_df: pd.DataFrame,
    sensor_cols: List[str],
    periods: List[int],
    output_dir: Path,
    top_n: int = 10,
    dpi: int = 150,
) -> Dict[str, Path]:
    """Create all report figures and save to disk.

    Args:
        df: DataFrame with sensor data.
        scores_df: DataFrame with scores.
        sensor_cols: List of sensor columns.
        periods: Target periods.
        output_dir: Output directory.
        top_n: Number of top sensors for detailed reports.
        dpi: Figure resolution.

    Returns:
        Dictionary mapping figure names to paths.
    """
    from seasonality.visualization.summary import (
        create_thumbnail_grid,
        create_detailed_report,
        plot_confidence_distribution,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_figures = {}

    # 1. Thumbnail grid
    try:
        fig = create_thumbnail_grid(df, sensor_cols, scores_df)
        path = output_dir / "thumbnail_grid.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved_figures["thumbnail_grid"] = path
        logger.info(f"Saved: {path}")
    except Exception as e:
        logger.warning(f"Failed to create thumbnail grid: {e}")

    # 2. Confidence distribution
    try:
        fig = plot_confidence_distribution(scores_df, mode="strict")
        path = output_dir / "confidence_distribution.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved_figures["confidence_distribution"] = path
        logger.info(f"Saved: {path}")
    except Exception as e:
        logger.warning(f"Failed to create confidence distribution: {e}")

    # 3. Detailed reports for top sensors
    if "composite_score" in scores_df.columns:
        top_sensors = scores_df.nlargest(top_n, "composite_score").index.tolist()
    else:
        top_sensors = sensor_cols[:top_n]

    for sensor in top_sensors:
        try:
            scores = scores_df.loc[sensor].to_dict() if sensor in scores_df.index else None
            fig = create_detailed_report(df, sensor, periods, scores)
            path = output_dir / f"detailed_{sensor}.png"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            saved_figures[f"detailed_{sensor}"] = path
            logger.info(f"Saved: {path}")
        except Exception as e:
            logger.warning(f"Failed to create detailed report for {sensor}: {e}")

    return saved_figures


def generate_html_report(
    scores_df: pd.DataFrame,
    summary_stats: Dict[str, Any],
    figures: Optional[Dict[str, Path]] = None,
    output_path: Optional[Path] = None,
    title: str = "Seasonality Analysis Report",
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate complete HTML report.

    Args:
        scores_df: DataFrame with scores.
        summary_stats: Summary statistics dictionary.
        figures: Dictionary of figure paths.
        output_path: Path to save HTML file.
        title: Report title.
        config: Configuration used for analysis.

    Returns:
        HTML content string.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='ja'>",
        "<head>",
        "  <meta charset='UTF-8'>",
        "  <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        f"  <title>{title}</title>",
        "  <style>",
        _get_report_css(),
        "  </style>",
        "</head>",
        "<body>",
        f"  <h1>{title}</h1>",
        f"  <p class='timestamp'>Generated: {timestamp}</p>",
    ]

    # Configuration section
    if config:
        html_parts.append("  <section class='config'>")
        html_parts.append("    <h2>Configuration</h2>")
        html_parts.append("    <pre>" + _format_config(config) + "</pre>")
        html_parts.append("  </section>")

    # Summary statistics section
    html_parts.append("  <section class='summary'>")
    html_parts.append("    <h2>Summary Statistics</h2>")
    html_parts.append(_format_summary_stats(summary_stats))
    html_parts.append("  </section>")

    # Figures section
    if figures:
        html_parts.append("  <section class='figures'>")
        html_parts.append("    <h2>Visualizations</h2>")

        # Thumbnail grid first
        if "thumbnail_grid" in figures:
            html_parts.append("    <div class='figure-container'>")
            html_parts.append("      <h3>Sensor Overview</h3>")
            html_parts.append(f"      <img src='{figures['thumbnail_grid'].name}' alt='Thumbnail Grid'>")
            html_parts.append("    </div>")

        # Confidence distribution
        if "confidence_distribution" in figures:
            html_parts.append("    <div class='figure-container'>")
            html_parts.append("      <h3>Confidence Distribution</h3>")
            html_parts.append(f"      <img src='{figures['confidence_distribution'].name}' alt='Confidence Distribution'>")
            html_parts.append("    </div>")

        # Detailed reports
        detailed_figs = [k for k in figures if k.startswith("detailed_")]
        if detailed_figs:
            html_parts.append("    <h3>Detailed Sensor Reports</h3>")
            html_parts.append("    <div class='detailed-reports'>")
            for fig_key in detailed_figs:
                sensor = fig_key.replace("detailed_", "")
                html_parts.append(f"      <div class='figure-container'>")
                html_parts.append(f"        <h4>{sensor}</h4>")
                html_parts.append(f"        <img src='{figures[fig_key].name}' alt='{sensor}'>")
                html_parts.append(f"      </div>")
            html_parts.append("    </div>")

        html_parts.append("  </section>")

    # Scores table section
    html_parts.append("  <section class='scores'>")
    html_parts.append("    <h2>Seasonality Scores</h2>")
    html_parts.append(_dataframe_to_html(scores_df))
    html_parts.append("  </section>")

    html_parts.extend([
        "  <footer>",
        "    <p>Generated by Seasonality Analyzer</p>",
        "  </footer>",
        "</body>",
        "</html>",
    ])

    html_content = "\n".join(html_parts)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"Saved HTML report: {output_path}")

    return html_content


def _get_report_css() -> str:
    """Get CSS styles for HTML report."""
    return """
    * {
      box-sizing: border-box;
    }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
      max-width: 1400px;
      margin: 0 auto;
      padding: 20px;
      background-color: #f5f5f5;
      color: #333;
    }
    h1 {
      color: #2c3e50;
      border-bottom: 3px solid #3498db;
      padding-bottom: 15px;
    }
    h2 {
      color: #34495e;
      margin-top: 40px;
      border-left: 4px solid #3498db;
      padding-left: 10px;
    }
    h3 {
      color: #555;
    }
    .timestamp {
      color: #888;
      font-size: 0.9em;
    }
    section {
      background: white;
      padding: 20px;
      margin: 20px 0;
      border-radius: 8px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin: 20px 0;
      font-size: 0.9em;
    }
    th, td {
      padding: 10px 8px;
      text-align: left;
      border-bottom: 1px solid #ddd;
    }
    th {
      background-color: #3498db;
      color: white;
      position: sticky;
      top: 0;
    }
    tr:hover {
      background-color: #f5f5f5;
    }
    .figure-container {
      margin: 20px 0;
      text-align: center;
    }
    .figure-container img {
      max-width: 100%;
      height: auto;
      border: 1px solid #ddd;
      border-radius: 4px;
    }
    .detailed-reports {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(800px, 1fr));
      gap: 20px;
    }
    pre {
      background: #f8f9fa;
      padding: 15px;
      border-radius: 4px;
      overflow-x: auto;
      font-size: 0.85em;
    }
    .high { color: #27ae60; font-weight: bold; }
    .medium { color: #f39c12; font-weight: bold; }
    .low { color: #3498db; }
    .none { color: #95a5a6; }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 15px;
    }
    .summary-card {
      background: #f8f9fa;
      padding: 15px;
      border-radius: 4px;
      text-align: center;
    }
    .summary-card .value {
      font-size: 2em;
      font-weight: bold;
      color: #3498db;
    }
    .summary-card .label {
      color: #666;
      font-size: 0.9em;
    }
    footer {
      text-align: center;
      color: #888;
      margin-top: 40px;
      padding: 20px;
    }
    """


def _format_config(config: Dict[str, Any]) -> str:
    """Format configuration for display."""
    import json
    return json.dumps(config, indent=2, ensure_ascii=False, default=str)


def _format_summary_stats(stats: Dict[str, Any]) -> str:
    """Format summary statistics as HTML."""
    html = ["<div class='summary-grid'>"]

    # Total sensors
    html.append(f"""
    <div class='summary-card'>
      <div class='value'>{stats.get('total_sensors', 'N/A')}</div>
      <div class='label'>Total Sensors</div>
    </div>
    """)

    # Confidence distribution
    for mode in ["strict", "exploratory"]:
        dist_key = f"confidence_distribution_{mode}"
        if dist_key in stats:
            dist = stats[dist_key]
            high = dist.get("High", 0)
            medium = dist.get("Medium", 0)
            low = dist.get("Low", 0)
            html.append(f"""
            <div class='summary-card'>
              <div class='value'>
                <span class='high'>{high}</span> /
                <span class='medium'>{medium}</span> /
                <span class='low'>{low}</span>
              </div>
              <div class='label'>H/M/L ({mode.capitalize()})</div>
            </div>
            """)

    # Composite score stats
    if "composite_score_stats" in stats:
        cs = stats["composite_score_stats"]
        html.append(f"""
        <div class='summary-card'>
          <div class='value'>{cs.get('max', 0):.3f}</div>
          <div class='label'>Max Composite Score</div>
        </div>
        <div class='summary-card'>
          <div class='value'>{cs.get('mean', 0):.3f}</div>
          <div class='label'>Mean Composite Score</div>
        </div>
        """)

    html.append("</div>")
    return "\n".join(html)


def _dataframe_to_html(df: pd.DataFrame, max_rows: int = 100) -> str:
    """Convert DataFrame to HTML table."""
    # Select display columns
    priority_cols = [
        "rank", "composite_score", "confidence_strict", "confidence_exploratory",
        "stl_max", "acf_max", "fourier_max", "periodogram_max",
        "best_period",
    ]
    display_cols = [c for c in priority_cols if c in df.columns]

    # Add remaining columns
    for col in df.columns:
        if col not in display_cols and len(display_cols) < 15:
            display_cols.append(col)

    html = ["<div style='overflow-x: auto;'>", "<table>", "<thead><tr><th>Sensor</th>"]
    for col in display_cols:
        html.append(f"<th>{col}</th>")
    html.append("</tr></thead>")

    html.append("<tbody>")
    for idx, row in df.head(max_rows).iterrows():
        html.append(f"<tr><td>{idx}</td>")
        for col in display_cols:
            value = row.get(col, "")
            css_class = ""

            if isinstance(value, float):
                value = f"{value:.4f}"
            elif col in ["confidence_strict", "confidence_exploratory"]:
                css_class = f" class='{str(value).lower()}'"

            html.append(f"<td{css_class}>{value}</td>")
        html.append("</tr>")
    html.append("</tbody></table>")

    if len(df) > max_rows:
        html.append(f"<p><em>Showing first {max_rows} of {len(df)} sensors</em></p>")

    html.append("</div>")
    return "\n".join(html)
