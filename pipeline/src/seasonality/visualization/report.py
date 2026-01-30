"""季節性分析のためのHTMLレポート生成モジュール。"""

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
    """matplotlibの図をbase64エンコード文字列に変換します。

    Args:
        fig: Matplotlibの図。
        format: 画像形式（'png'または'svg'）。

    Returns:
        Base64エンコードされた文字列。
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
    """全てのレポート図を作成してディスクに保存します。

    Args:
        df: センサーデータを含むDataFrame。
        scores_df: スコアを含むDataFrame。
        sensor_cols: センサー列のリスト。
        periods: ターゲット周期。
        output_dir: 出力ディレクトリ。
        top_n: 詳細レポートを作成する上位センサーの数。
        dpi: 図の解像度。

    Returns:
        図の名前からパスへのマッピング辞書。
    """
    from seasonality.visualization.summary import (
        create_thumbnail_grid,
        create_detailed_report,
        plot_confidence_distribution,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_figures = {}

    # 1. サムネイルグリッド
    try:
        fig = create_thumbnail_grid(df, sensor_cols, scores_df)
        path = output_dir / "thumbnail_grid.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved_figures["thumbnail_grid"] = path
        logger.info(f"保存しました: {path}")
    except Exception as e:
        logger.warning(f"サムネイルグリッドの作成に失敗しました: {e}")

    # 2. 信頼度分布
    try:
        fig = plot_confidence_distribution(scores_df, mode="strict")
        path = output_dir / "confidence_distribution.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved_figures["confidence_distribution"] = path
        logger.info(f"保存しました: {path}")
    except Exception as e:
        logger.warning(f"信頼度分布の作成に失敗しました: {e}")

    # 3. 上位センサーの詳細レポート
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
            logger.info(f"保存しました: {path}")
        except Exception as e:
            logger.warning(f"{sensor}の詳細レポート作成に失敗しました: {e}")

    return saved_figures


def generate_html_report(
    scores_df: pd.DataFrame,
    summary_stats: Dict[str, Any],
    figures: Optional[Dict[str, Path]] = None,
    output_path: Optional[Path] = None,
    title: str = "季節性分析レポート",
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """完全なHTMLレポートを生成します。

    Args:
        scores_df: スコアを含むDataFrame。
        summary_stats: 要約統計の辞書。
        figures: 図のパスの辞書。
        output_path: HTMLファイルを保存するパス。
        title: レポートのタイトル。
        config: 分析に使用した設定。

    Returns:
        HTMLコンテンツ文字列。
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
        f"  <p class='timestamp'>生成日時: {timestamp}</p>",
    ]

    # 設定セクション
    if config:
        html_parts.append("  <section class='config'>")
        html_parts.append("    <h2>設定</h2>")
        html_parts.append("    <pre>" + _format_config(config) + "</pre>")
        html_parts.append("  </section>")

    # 要約統計セクション
    html_parts.append("  <section class='summary'>")
    html_parts.append("    <h2>要約統計</h2>")
    html_parts.append(_format_summary_stats(summary_stats))
    html_parts.append("  </section>")

    # 図のセクション
    if figures:
        html_parts.append("  <section class='figures'>")
        html_parts.append("    <h2>可視化</h2>")

        # サムネイルグリッドを最初に
        if "thumbnail_grid" in figures:
            html_parts.append("    <div class='figure-container'>")
            html_parts.append("      <h3>センサー概要</h3>")
            html_parts.append(f"      <img src='{figures['thumbnail_grid'].name}' alt='サムネイルグリッド'>")
            html_parts.append("    </div>")

        # 信頼度分布
        if "confidence_distribution" in figures:
            html_parts.append("    <div class='figure-container'>")
            html_parts.append("      <h3>信頼度分布</h3>")
            html_parts.append(f"      <img src='{figures['confidence_distribution'].name}' alt='信頼度分布'>")
            html_parts.append("    </div>")

        # 詳細レポート
        detailed_figs = [k for k in figures if k.startswith("detailed_")]
        if detailed_figs:
            html_parts.append("    <h3>詳細センサーレポート</h3>")
            html_parts.append("    <div class='detailed-reports'>")
            for fig_key in detailed_figs:
                sensor = fig_key.replace("detailed_", "")
                html_parts.append(f"      <div class='figure-container'>")
                html_parts.append(f"        <h4>{sensor}</h4>")
                html_parts.append(f"        <img src='{figures[fig_key].name}' alt='{sensor}'>")
                html_parts.append(f"      </div>")
            html_parts.append("    </div>")

        html_parts.append("  </section>")

    # スコアテーブルセクション
    html_parts.append("  <section class='scores'>")
    html_parts.append("    <h2>季節性スコア</h2>")
    html_parts.append(_dataframe_to_html(scores_df))
    html_parts.append("  </section>")

    html_parts.extend([
        "  <footer>",
        "    <p>Seasonality Analyzerにより生成</p>",
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
        logger.info(f"HTMLレポートを保存しました: {output_path}")

    return html_content


def _get_report_css() -> str:
    """HTMLレポート用のCSSスタイルを取得します。"""
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
    """表示用に設定をフォーマットします。"""
    import json
    return json.dumps(config, indent=2, ensure_ascii=False, default=str)


def _format_summary_stats(stats: Dict[str, Any]) -> str:
    """要約統計をHTMLとしてフォーマットします。"""
    html = ["<div class='summary-grid'>"]

    # 総センサー数
    html.append(f"""
    <div class='summary-card'>
      <div class='value'>{stats.get('total_sensors', 'N/A')}</div>
      <div class='label'>総センサー数</div>
    </div>
    """)

    # 信頼度分布
    for mode in ["strict", "exploratory"]:
        dist_key = f"confidence_distribution_{mode}"
        if dist_key in stats:
            dist = stats[dist_key]
            high = dist.get("High", 0)
            medium = dist.get("Medium", 0)
            low = dist.get("Low", 0)
            mode_label = "厳格" if mode == "strict" else "探索的"
            html.append(f"""
            <div class='summary-card'>
              <div class='value'>
                <span class='high'>{high}</span> /
                <span class='medium'>{medium}</span> /
                <span class='low'>{low}</span>
              </div>
              <div class='label'>高/中/低 ({mode_label})</div>
            </div>
            """)

    # 総合スコア統計
    if "composite_score_stats" in stats:
        cs = stats["composite_score_stats"]
        html.append(f"""
        <div class='summary-card'>
          <div class='value'>{cs.get('max', 0):.3f}</div>
          <div class='label'>最大総合スコア</div>
        </div>
        <div class='summary-card'>
          <div class='value'>{cs.get('mean', 0):.3f}</div>
          <div class='label'>平均総合スコア</div>
        </div>
        """)

    html.append("</div>")
    return "\n".join(html)


def _dataframe_to_html(df: pd.DataFrame, max_rows: int = 100) -> str:
    """DataFrameをHTMLテーブルに変換します。"""
    # 表示列を選択
    priority_cols = [
        "rank", "composite_score", "confidence_strict", "confidence_exploratory",
        "stl_max", "acf_max", "fourier_max", "periodogram_max",
        "best_period",
    ]
    display_cols = [c for c in priority_cols if c in df.columns]

    # 残りの列を追加
    for col in df.columns:
        if col not in display_cols and len(display_cols) < 15:
            display_cols.append(col)

    html = ["<div style='overflow-x: auto;'>", "<table>", "<thead><tr><th>センサー</th>"]
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
        html.append(f"<p><em>{len(df)}センサーのうち最初の{max_rows}件を表示</em></p>")

    html.append("</div>")
    return "\n".join(html)
