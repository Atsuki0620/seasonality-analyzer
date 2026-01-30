"""Seasonality AnalyzerのCLIインターフェース"""

from __future__ import annotations

import gc
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from loguru import logger

from seasonality import __version__


@click.group()
@click.version_option(version=__version__)
def main():
    """Seasonality Analyzer - 時系列データの季節性パターンを検出"""
    pass


@main.command()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="設定YAMLファイルのパス",
)
@click.option(
    "--data", "-d",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="入力CSVデータファイルのパス",
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=Path("outputs"),
    help="出力ディレクトリ",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="ログレベル",
)
@click.option(
    "--no-figures",
    is_flag=True,
    help="図の生成をスキップ",
)
@click.option(
    "--sensors",
    type=str,
    default=None,
    help="分析するセンサー名のカンマ区切りリスト（デフォルト: すべて）",
)
def run(
    config: Optional[Path],
    data: Path,
    output: Path,
    log_level: str,
    no_figures: bool,
    sensors: Optional[str],
):
    """季節性分析パイプラインを実行"""
    from seasonality.config import SeasonalityConfig, load_config
    from seasonality.io.loader import load_sensor_data, validate_data_for_analysis
    from seasonality.io.writer import ensure_output_dirs, save_results
    from seasonality.preprocessing.resample import resample_and_interpolate
    from seasonality.detection.ensemble import EnsembleDetector
    from seasonality.scoring.ranking import create_scores_dataframe, generate_ranking, generate_summary_statistics
    from seasonality.visualization.report import create_report_figures, generate_html_report
    from seasonality.utils.logging import setup_logger, shutdown_logger
    from seasonality.utils.profiling import StepTimer, MemoryTracker
    from seasonality.debug.bundler import DebugBundleBuilder, save_debug_bundle

    sinks: dict[str, int] = {}

    # ログ設定
    output = Path(output)
    log_file = output / "logs" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    sinks = setup_logger(log_file=log_file, level=log_level)

    logger.info(f"Seasonality Analyzer v{__version__}")
    logger.info(f"Data file: {data}")
    logger.info(f"Config file: {config or 'default'}")

    timer = StepTimer()
    memory = MemoryTracker("main")
    memory.start()

    debug_builder = DebugBundleBuilder(version=__version__)
    errors = []

    try:
        # 設定読み込み
        with timer.step("load_config"):
            cfg = load_config(config)
            cfg.output.base_dir = Path(output)
            # PathオブジェクトをS文字列に変換して設定を辞書化
            config_dict = cfg.model_dump()
            config_dict["output"]["base_dir"] = str(cfg.output.base_dir)
            debug_builder.set_config(config_dict)
            logger.info(f"Scoring mode: {cfg.scoring.mode}")
            logger.info(f"Target periods: {cfg.detection.periods}")

        # 出力ディレクトリ作成
        dirs = ensure_output_dirs(cfg.output)

        # データ読み込み
        with timer.step("load_data"):
            df, health_report = load_sensor_data(data, cfg.data)
            debug_builder.set_data_summary(
                total_records=health_report.total_records,
                date_range=health_report.date_range,
                sensor_count=health_report.sensor_count,
                missing_rate=health_report.missing_rate,
                sensor_columns=health_report.sensor_columns,
            )
            debug_builder.add_health_check(
                "data_load",
                health_report.is_healthy(),
                health_report.summary(),
            )

        # 分析対象センサーの決定
        if sensors:
            sensor_cols = [s.strip() for s in sensors.split(",")]
            missing = [s for s in sensor_cols if s not in health_report.sensor_columns]
            if missing:
                logger.warning(f"センサーが見つかりません: {missing}")
                sensor_cols = [s for s in sensor_cols if s in health_report.sensor_columns]
        else:
            sensor_cols = health_report.sensor_columns

        # センサー検証
        valid_sensors, skipped = validate_data_for_analysis(df, sensor_cols)
        logger.info(f"{len(valid_sensors)}個のセンサーを分析中（スキップ: {len(skipped)}個）")

        if not valid_sensors:
            raise ValueError("分析可能な有効なセンサーがありません")

        # データ前処理
        with timer.step("preprocessing"):
            import pandas as pd
            # 最初の有効なセンサーからリサンプルインデックスを作成
            first_sensor_resampled = resample_and_interpolate(
                df[valid_sensors[0]],
                resample_freq=cfg.preprocessing.resample_freq,
                resample_method=cfg.preprocessing.resample_method,
                interp_method=cfg.preprocessing.impute_method if cfg.preprocessing.impute_method != "none" else None,
            )
            processed_df = pd.DataFrame(index=first_sensor_resampled.index)
            processed_df[valid_sensors[0]] = first_sensor_resampled

            for sensor in valid_sensors[1:]:
                processed_df[sensor] = resample_and_interpolate(
                    df[sensor],
                    resample_freq=cfg.preprocessing.resample_freq,
                    resample_method=cfg.preprocessing.resample_method,
                    interp_method=cfg.preprocessing.impute_method if cfg.preprocessing.impute_method != "none" else None,
                )

            debug_builder.add_health_check(
                "preprocessing",
                True,
                f"Preprocessed {len(valid_sensors)} sensors",
            )

        memory.checkpoint("after_preprocessing")

        # 季節性検出
        with timer.step("detection"):
            detector = EnsembleDetector(config=cfg.detection)
            ensemble_results = detector.detect_batch(
                processed_df,
                valid_sensors,
                periods=cfg.detection.periods,
            )

            debug_builder.add_health_check(
                "detection",
                len(ensemble_results) > 0,
                f"Detected seasonality in {len(ensemble_results)} sensors",
            )

        memory.checkpoint("after_detection")

        # スコアリングとランキング
        with timer.step("scoring"):
            scores_df = create_scores_dataframe(ensemble_results, cfg.scoring)
            ranking_df = generate_ranking(scores_df)
            summary_stats = generate_summary_statistics(scores_df, cfg.scoring)

            debug_builder.set_results_summary(summary_stats)
            debug_builder.add_health_check(
                "scoring",
                True,
                f"Scored {len(scores_df)} sensors",
            )

        # 結果保存
        with timer.step("save_results"):
            results = {
                "scores": scores_df,
                "ranking": ranking_df,
                "summary": summary_stats,
                "config": cfg.model_dump(),
            }
            saved_files = save_results(results, dirs["results"], prefix="seasonality")

        # 図の生成
        if not no_figures:
            with timer.step("visualization"):
                try:
                    import matplotlib
                    matplotlib.use("Agg")  # 非対話的バックエンド

                    # 日本語フォント設定を初期化
                    from seasonality.utils.japanese_font import setup_japanese_font
                    setup_japanese_font()

                    figures = create_report_figures(
                        df=processed_df,
                        scores_df=ranking_df,
                        sensor_cols=valid_sensors,
                        periods=cfg.detection.periods,
                        output_dir=dirs["figures"],
                        top_n=10,
                        dpi=cfg.output.figure_dpi,
                    )

                    # HTMLレポート生成
                    html_path = dirs["results"] / "report.html"
                    generate_html_report(
                        scores_df=ranking_df,
                        summary_stats=summary_stats,
                        figures=figures,
                        output_path=html_path,
                        config=cfg.model_dump(),
                    )
                except Exception as e:
                    logger.warning(f"可視化に失敗しました: {e}")
                    errors.append({
                        "error_type": "VisualizationError",
                        "message": str(e),
                    })

        memory.checkpoint("after_visualization")

        # サマリー表示
        timer.print_summary()
        memory.stop()

        click.echo("\n=== 分析完了 ===")
        click.echo(f"分析したセンサー総数: {len(scores_df)}")
        click.echo(f"信頼度分布（厳密モード）:")
        for level, count in summary_stats.get("confidence_distribution_strict", {}).items():
            click.echo(f"  {level}: {count}")
        click.echo(f"\n結果保存先: {dirs['results']}")
        click.echo(f"図の保存先: {dirs['figures']}")
        click.echo(f"ログ保存先: {log_file}")

        # デバッグバンドル保存
        for step in timer.steps:
            debug_builder.add_processing_step(
                name=step.name,
                start_time=step.start_time,
                end_time=step.end_time,
                status="success",
            )

        bundle = debug_builder.build()
        bundle_path = dirs["debug_bundles"] / f"bundle_{bundle.bundle_id}.json"
        save_debug_bundle(bundle, bundle_path)

    except Exception as e:
        logger.exception(f"パイプライン実行に失敗しました: {e}")
        errors.append({
            "error_type": type(e).__name__,
            "message": str(e),
        })
        for err in errors:
            debug_builder.add_error(**err)
        bundle = debug_builder.build()
        bundle_path = output / "debug_bundles" / f"bundle_{bundle.bundle_id}.json"
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        save_debug_bundle(bundle, bundle_path)
        click.echo(f"エラー: {e}", err=True)
        click.echo(f"デバッグバンドル保存先: {bundle_path}", err=True)
        sys.exit(1)

    finally:
        shutdown_logger(sinks)
        gc.collect()


@main.command()
@click.option(
    "--log",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="ログファイルのパス",
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=Path("outputs/debug_bundles"),
    help="デバッグバンドルの出力ディレクトリ",
)
def debug(log: Path, output: Path):
    """ログファイルからデバッグバンドルを生成"""
    from seasonality.debug.bundler import DebugBundleBuilder, save_debug_bundle

    click.echo(f"デバッグバンドル生成中: {log}")

    builder = DebugBundleBuilder()

    # ログファイル読み込み
    with open(log, "r", encoding="utf-8") as f:
        log_content = f.read()

    builder.set_log_excerpt(log_content, max_lines=200)
    builder.set_config({"source": "log_file", "log_path": str(log)})
    builder.set_data_summary(
        total_records=0,
        date_range=("unknown", "unknown"),
        sensor_count=0,
        missing_rate=0,
    )

    # ログからエラーを抽出
    import re
    error_pattern = r"ERROR.*?:(.*?)(?=\d{4}-\d{2}-\d{2}|$)"
    errors = re.findall(error_pattern, log_content, re.DOTALL)
    for error in errors[:10]:  # 最大10個のエラーに制限
        builder.add_error("LogError", error.strip()[:500])

    bundle = builder.build()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    bundle_path = output / f"bundle_{bundle.bundle_id}.json"
    save_debug_bundle(bundle, bundle_path, anonymize=True)

    click.echo(f"デバッグバンドル保存先: {bundle_path}")


@main.command()
@click.option(
    "--bundle", "-b",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="デバッグバンドルJSONファイルのパス",
)
def diagnose(bundle: Path):
    """デバッグバンドルに対してLLMベースの診断を実行"""
    from seasonality.debug.bundler import load_debug_bundle
    from seasonality.debug.llm_client import create_diagnostic_client

    click.echo(f"デバッグバンドル読み込み中: {bundle}")
    debug_bundle = load_debug_bundle(bundle)

    click.echo("診断分析を実行中...")
    client = create_diagnostic_client()

    if not client.is_available:
        click.echo("警告: LLMプロバイダーが設定されていません。OPENAI_API_KEYまたはAzure OpenAI認証情報を設定してください。", err=True)
        click.echo("\n基本分析（LLMなし）:")
        click.echo(f"  バンドルID: {debug_bundle.bundle_id}")
        click.echo(f"  作成日時: {debug_bundle.created_at}")
        click.echo(f"  エラー数: {len(debug_bundle.errors)}")
        click.echo(f"  警告数: {len(debug_bundle.warnings)}")

        if debug_bundle.errors:
            click.echo("\n検出されたエラー:")
            for err in debug_bundle.errors[:5]:
                click.echo(f"  - [{err.get('type', 'Unknown')}] {err.get('message', '')[:100]}")
        return

    result = client.diagnose(debug_bundle)

    click.echo(f"\n=== 診断結果 ===")
    click.echo(f"信頼度: {result.confidence}")
    click.echo(f"\n要約:\n{result.summary}")

    if result.issues_found:
        click.echo(f"\n検出された問題 ({len(result.issues_found)}件):")
        for issue in result.issues_found:
            severity = issue.get("severity", "unknown")
            category = issue.get("category", "unknown")
            desc = issue.get("description", "No description")
            click.echo(f"  [{severity.upper()}] ({category}) {desc}")

    if result.recommendations:
        click.echo(f"\n推奨事項:")
        for rec in result.recommendations:
            click.echo(f"  - {rec}")


@main.command()
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=Path("config"),
    help="設定ファイルの出力ディレクトリ",
)
def init(output: Path):
    """設定ファイルを初期化"""
    from seasonality.config import SeasonalityConfig

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    # デフォルト設定を作成
    default_config = SeasonalityConfig()
    default_path = output / "default.yaml"
    default_config.to_yaml(default_path)
    click.echo(f"Created: {default_path}")

    # 厳密モード設定を作成
    from seasonality.config import ScoringConfig
    strict_config = SeasonalityConfig(
        scoring=ScoringConfig(mode="strict")
    )
    strict_path = output / "strict_mode.yaml"
    strict_config.to_yaml(strict_path)
    click.echo(f"Created: {strict_path}")

    # 探索モード設定を作成
    exploratory_config = SeasonalityConfig(
        scoring=ScoringConfig(mode="exploratory")
    )
    exploratory_path = output / "exploratory_mode.yaml"
    exploratory_config.to_yaml(exploratory_path)
    click.echo(f"Created: {exploratory_path}")

    click.echo(f"\n設定ファイル作成先: {output}")


if __name__ == "__main__":
    main()
