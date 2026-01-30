"""Seasonality Analyzerの出力書き込みユーティリティ"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from seasonality.config import OutputConfig


def ensure_output_dirs(config: OutputConfig) -> Dict[str, Path]:
    """すべての出力ディレクトリが存在することを確認

    Args:
        config: 出力設定

    Returns:
        ディレクトリ名からパスへのマッピング辞書
    """
    dirs = {
        "results": config.base_dir / config.results_dir,
        "figures": config.base_dir / config.figures_dir,
        "logs": config.base_dir / config.logs_dir,
        "debug_bundles": config.base_dir / config.debug_bundles_dir,
    }

    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"ディレクトリを確保しました: {path}")

    return dirs


def save_scores_csv(
    scores_df: pd.DataFrame,
    output_path: Path | str,
    include_timestamp: bool = True,
) -> Path:
    """スコアDataFrameをCSVに保存

    Args:
        scores_df: 季節性スコアを含むDataFrame
        output_path: 出力ファイルパス
        include_timestamp: ファイル名にタイムスタンプを含めるか

    Returns:
        保存されたファイルのパス
    """
    output_path = Path(output_path)

    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = output_path.stem
        suffix = output_path.suffix
        output_path = output_path.parent / f"{stem}_{timestamp}{suffix}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(output_path)
    logger.info(f"スコアを保存しました: {output_path}")

    return output_path


def save_results(
    results: Dict[str, Any],
    output_dir: Path | str,
    prefix: str = "results",
) -> Dict[str, Path]:
    """分析結果を複数のファイルに保存

    Args:
        results: 分析結果を含む辞書
        output_dir: 出力ディレクトリ
        prefix: ファイル名プレフィックス

    Returns:
        結果タイプからファイルパスへのマッピング辞書
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # スコアが存在する場合は保存
    if "scores" in results and isinstance(results["scores"], pd.DataFrame):
        scores_path = output_dir / f"{prefix}_scores_{timestamp}.csv"
        results["scores"].to_csv(scores_path)
        saved_files["scores"] = scores_path
        logger.info(f"スコアを保存しました: {scores_path}")

    # ランキングが存在する場合は保存
    if "ranking" in results and isinstance(results["ranking"], pd.DataFrame):
        ranking_path = output_dir / f"{prefix}_ranking_{timestamp}.csv"
        results["ranking"].to_csv(ranking_path)
        saved_files["ranking"] = ranking_path
        logger.info(f"ランキングを保存しました: {ranking_path}")

    # サマリー統計が存在する場合は保存
    if "summary" in results:
        summary_path = output_dir / f"{prefix}_summary_{timestamp}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(_make_json_serializable(results["summary"]), f, indent=2, ensure_ascii=False)
        saved_files["summary"] = summary_path
        logger.info(f"サマリーを保存しました: {summary_path}")

    # 設定が存在する場合は保存
    if "config" in results:
        config_path = output_dir / f"{prefix}_config_{timestamp}.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(_make_json_serializable(results["config"]), f, indent=2, ensure_ascii=False)
        saved_files["config"] = config_path
        logger.info(f"設定を保存しました: {config_path}")

    return saved_files


def _make_json_serializable(obj: Any) -> Any:
    """オブジェクトをJSONシリアライズ可能な形式に変換"""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    elif isinstance(obj, Path):
        return str(obj)
    elif hasattr(obj, "model_dump"):  # Pydanticモデル
        return obj.model_dump()
    elif hasattr(obj, "__dict__"):
        return _make_json_serializable(obj.__dict__)
    else:
        return obj


def export_html_report(
    scores_df: pd.DataFrame,
    output_path: Path | str,
    title: str = "季節性分析レポート",
    figures: Optional[List[Path]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """分析結果をHTMLレポートとしてエクスポート

    Args:
        scores_df: 季節性スコアを含むDataFrame
        output_path: 出力ファイルパス
        title: レポートタイトル
        figures: 埋め込む図のパスのリスト
        metadata: 含める追加メタデータ

    Returns:
        保存されたHTMLファイルのパス
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # HTMLコンテンツ生成
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
        f"  <p class='timestamp'>生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
    ]

    # メタデータセクション追加
    if metadata:
        html_parts.append("  <section class='metadata'>")
        html_parts.append("    <h2>分析設定</h2>")
        html_parts.append("    <dl>")
        for key, value in metadata.items():
            html_parts.append(f"      <dt>{key}</dt>")
            html_parts.append(f"      <dd>{value}</dd>")
        html_parts.append("    </dl>")
        html_parts.append("  </section>")

    # スコア表追加
    html_parts.append("  <section class='scores'>")
    html_parts.append("    <h2>季節性スコア</h2>")
    html_parts.append(_dataframe_to_html(scores_df))
    html_parts.append("  </section>")

    # 図を追加
    if figures:
        html_parts.append("  <section class='figures'>")
        html_parts.append("    <h2>可視化</h2>")
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

    # HTMLファイル書き込み
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    logger.info(f"HTMLレポートをエクスポートしました: {output_path}")
    return output_path


def _get_report_css() -> str:
    """HTMLレポート用のCSSスタイルを取得"""
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
    """DataFrameをスタイル付きHTML表に変換"""
    # 表示列を選択
    display_cols = []
    priority_cols = ["composite_score", "confidence_strict", "confidence_exploratory",
                     "stl_max", "acf_max", "fourier_full"]
    for col in priority_cols:
        if col in df.columns:
            display_cols.append(col)
    # 残りの列を追加
    for col in df.columns:
        if col not in display_cols:
            display_cols.append(col)

    # HTML表を構築
    html = ["<table>", "<thead><tr><th>センサー</th>"]
    for col in display_cols[:10]:  # 列数を制限
        html.append(f"<th>{col}</th>")
    html.append("</tr></thead>")

    html.append("<tbody>")
    for idx, row in df.head(50).iterrows():  # 行数を制限
        html.append(f"<tr><td>{idx}</td>")
        for col in display_cols[:10]:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            # 信頼度クラスを追加
            css_class = ""
            if col in ["confidence_strict", "confidence_exploratory"]:
                css_class = f" class='{str(value).lower()}'"
            html.append(f"<td{css_class}>{value}</td>")
        html.append("</tr>")
    html.append("</tbody></table>")

    if len(df) > 50:
        html.append(f"<p><em>{len(df)}センサー中、最初の50個を表示</em></p>")

    return "\n".join(html)
