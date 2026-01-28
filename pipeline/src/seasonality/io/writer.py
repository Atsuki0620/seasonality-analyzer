"""Output writing utilities for Seasonality Analyzer."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from seasonality.config import OutputConfig


def ensure_output_dirs(config: OutputConfig) -> Dict[str, Path]:
    """Ensure all output directories exist.

    Args:
        config: Output configuration.

    Returns:
        Dictionary mapping directory names to paths.
    """
    dirs = {
        "results": config.base_dir / config.results_dir,
        "figures": config.base_dir / config.figures_dir,
        "logs": config.base_dir / config.logs_dir,
        "debug_bundles": config.base_dir / config.debug_bundles_dir,
    }

    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory: {path}")

    return dirs


def save_scores_csv(
    scores_df: pd.DataFrame,
    output_path: Path | str,
    include_timestamp: bool = True,
) -> Path:
    """Save scores DataFrame to CSV.

    Args:
        scores_df: DataFrame containing seasonality scores.
        output_path: Output file path.
        include_timestamp: Whether to include timestamp in filename.

    Returns:
        Path to saved file.
    """
    output_path = Path(output_path)

    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = output_path.stem
        suffix = output_path.suffix
        output_path = output_path.parent / f"{stem}_{timestamp}{suffix}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(output_path)
    logger.info(f"Saved scores to: {output_path}")

    return output_path


def save_results(
    results: Dict[str, Any],
    output_dir: Path | str,
    prefix: str = "results",
) -> Dict[str, Path]:
    """Save analysis results to multiple files.

    Args:
        results: Dictionary containing analysis results.
        output_dir: Output directory.
        prefix: Filename prefix.

    Returns:
        Dictionary mapping result types to file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save scores if present
    if "scores" in results and isinstance(results["scores"], pd.DataFrame):
        scores_path = output_dir / f"{prefix}_scores_{timestamp}.csv"
        results["scores"].to_csv(scores_path)
        saved_files["scores"] = scores_path
        logger.info(f"Saved scores: {scores_path}")

    # Save ranking if present
    if "ranking" in results and isinstance(results["ranking"], pd.DataFrame):
        ranking_path = output_dir / f"{prefix}_ranking_{timestamp}.csv"
        results["ranking"].to_csv(ranking_path)
        saved_files["ranking"] = ranking_path
        logger.info(f"Saved ranking: {ranking_path}")

    # Save summary statistics if present
    if "summary" in results:
        summary_path = output_dir / f"{prefix}_summary_{timestamp}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(_make_json_serializable(results["summary"]), f, indent=2, ensure_ascii=False)
        saved_files["summary"] = summary_path
        logger.info(f"Saved summary: {summary_path}")

    # Save config if present
    if "config" in results:
        config_path = output_dir / f"{prefix}_config_{timestamp}.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(_make_json_serializable(results["config"]), f, indent=2, ensure_ascii=False)
        saved_files["config"] = config_path
        logger.info(f"Saved config: {config_path}")

    return saved_files


def _make_json_serializable(obj: Any) -> Any:
    """Convert object to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    elif isinstance(obj, Path):
        return str(obj)
    elif hasattr(obj, "model_dump"):  # Pydantic model
        return obj.model_dump()
    elif hasattr(obj, "__dict__"):
        return _make_json_serializable(obj.__dict__)
    else:
        return obj


def export_html_report(
    scores_df: pd.DataFrame,
    output_path: Path | str,
    title: str = "Seasonality Analysis Report",
    figures: Optional[List[Path]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Export analysis results as HTML report.

    Args:
        scores_df: DataFrame containing seasonality scores.
        output_path: Output file path.
        title: Report title.
        figures: List of figure paths to embed.
        metadata: Additional metadata to include.

    Returns:
        Path to saved HTML file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate HTML content
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
        f"  <p class='timestamp'>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
    ]

    # Add metadata section
    if metadata:
        html_parts.append("  <section class='metadata'>")
        html_parts.append("    <h2>Analysis Configuration</h2>")
        html_parts.append("    <dl>")
        for key, value in metadata.items():
            html_parts.append(f"      <dt>{key}</dt>")
            html_parts.append(f"      <dd>{value}</dd>")
        html_parts.append("    </dl>")
        html_parts.append("  </section>")

    # Add scores table
    html_parts.append("  <section class='scores'>")
    html_parts.append("    <h2>Seasonality Scores</h2>")
    html_parts.append(_dataframe_to_html(scores_df))
    html_parts.append("  </section>")

    # Add figures
    if figures:
        html_parts.append("  <section class='figures'>")
        html_parts.append("    <h2>Visualizations</h2>")
        for fig_path in figures:
            if fig_path.exists():
                html_parts.append(f"    <figure>")
                html_parts.append(f"      <img src='{fig_path.name}' alt='{fig_path.stem}'>")
                html_parts.append(f"      <figcaption>{fig_path.stem}</figcaption>")
                html_parts.append(f"    </figure>")
        html_parts.append("  </section>")

    html_parts.extend([
        "</body>",
        "</html>",
    ])

    # Write HTML file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    logger.info(f"Exported HTML report: {output_path}")
    return output_path


def _get_report_css() -> str:
    """Get CSS styles for HTML report."""
    return """
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
      max-width: 1200px;
      margin: 0 auto;
      padding: 20px;
      background-color: #f5f5f5;
    }
    h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
    h2 { color: #555; margin-top: 30px; }
    .timestamp { color: #888; font-size: 0.9em; }
    table {
      width: 100%;
      border-collapse: collapse;
      background: white;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
      margin: 20px 0;
    }
    th, td {
      padding: 12px;
      text-align: left;
      border-bottom: 1px solid #ddd;
    }
    th { background-color: #4CAF50; color: white; }
    tr:hover { background-color: #f5f5f5; }
    .metadata dl { display: grid; grid-template-columns: 200px 1fr; gap: 10px; }
    .metadata dt { font-weight: bold; color: #555; }
    .metadata dd { margin: 0; }
    .figures { display: flex; flex-wrap: wrap; gap: 20px; }
    figure { margin: 0; background: white; padding: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    figure img { max-width: 100%; height: auto; }
    figcaption { text-align: center; color: #666; font-size: 0.9em; margin-top: 10px; }
    .high { color: #4CAF50; font-weight: bold; }
    .medium { color: #FF9800; font-weight: bold; }
    .low { color: #2196F3; }
    .none { color: #9E9E9E; }
    """


def _dataframe_to_html(df: pd.DataFrame) -> str:
    """Convert DataFrame to styled HTML table."""
    # Select display columns
    display_cols = []
    priority_cols = ["composite_score", "confidence_strict", "confidence_exploratory",
                     "stl_max", "acf_max", "fourier_full"]
    for col in priority_cols:
        if col in df.columns:
            display_cols.append(col)
    # Add remaining columns
    for col in df.columns:
        if col not in display_cols:
            display_cols.append(col)

    # Build HTML table
    html = ["<table>", "<thead><tr><th>Sensor</th>"]
    for col in display_cols[:10]:  # Limit columns
        html.append(f"<th>{col}</th>")
    html.append("</tr></thead>")

    html.append("<tbody>")
    for idx, row in df.head(50).iterrows():  # Limit rows
        html.append(f"<tr><td>{idx}</td>")
        for col in display_cols[:10]:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            # Add confidence class
            css_class = ""
            if col in ["confidence_strict", "confidence_exploratory"]:
                css_class = f" class='{str(value).lower()}'"
            html.append(f"<td{css_class}>{value}</td>")
        html.append("</tr>")
    html.append("</tbody></table>")

    if len(df) > 50:
        html.append(f"<p><em>Showing first 50 of {len(df)} sensors</em></p>")

    return "\n".join(html)
