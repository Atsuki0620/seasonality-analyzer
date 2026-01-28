"""CLI interface for Seasonality Analyzer."""

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
    """Seasonality Analyzer - Detect seasonal patterns in time series data."""
    pass


@main.command()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to configuration YAML file",
)
@click.option(
    "--data", "-d",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to input CSV data file",
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=Path("outputs"),
    help="Output directory",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Logging level",
)
@click.option(
    "--no-figures",
    is_flag=True,
    help="Skip figure generation",
)
@click.option(
    "--sensors",
    type=str,
    default=None,
    help="Comma-separated list of sensor names to analyze (default: all)",
)
def run(
    config: Optional[Path],
    data: Path,
    output: Path,
    log_level: str,
    no_figures: bool,
    sensors: Optional[str],
):
    """Run seasonality analysis pipeline."""
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

    # Setup logging
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
        # Load configuration
        with timer.step("load_config"):
            cfg = load_config(config)
            cfg.output.base_dir = Path(output)
            # Convert config to dict with Path objects as strings
            config_dict = cfg.model_dump()
            config_dict["output"]["base_dir"] = str(cfg.output.base_dir)
            debug_builder.set_config(config_dict)
            logger.info(f"Scoring mode: {cfg.scoring.mode}")
            logger.info(f"Target periods: {cfg.detection.periods}")

        # Ensure output directories
        dirs = ensure_output_dirs(cfg.output)

        # Load data
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

        # Determine sensors to analyze
        if sensors:
            sensor_cols = [s.strip() for s in sensors.split(",")]
            missing = [s for s in sensor_cols if s not in health_report.sensor_columns]
            if missing:
                logger.warning(f"Sensors not found: {missing}")
                sensor_cols = [s for s in sensor_cols if s in health_report.sensor_columns]
        else:
            sensor_cols = health_report.sensor_columns

        # Validate sensors
        valid_sensors, skipped = validate_data_for_analysis(df, sensor_cols)
        logger.info(f"Analyzing {len(valid_sensors)} sensors (skipped {len(skipped)})")

        if not valid_sensors:
            raise ValueError("No valid sensors to analyze")

        # Preprocess data
        with timer.step("preprocessing"):
            import pandas as pd
            # Create resampled index from first valid sensor
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

        # Run detection
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

        # Score and rank
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

        # Save results
        with timer.step("save_results"):
            results = {
                "scores": scores_df,
                "ranking": ranking_df,
                "summary": summary_stats,
                "config": cfg.model_dump(),
            }
            saved_files = save_results(results, dirs["results"], prefix="seasonality")

        # Generate figures
        if not no_figures:
            with timer.step("visualization"):
                try:
                    import matplotlib
                    matplotlib.use("Agg")  # Non-interactive backend

                    figures = create_report_figures(
                        df=processed_df,
                        scores_df=ranking_df,
                        sensor_cols=valid_sensors,
                        periods=cfg.detection.periods,
                        output_dir=dirs["figures"],
                        top_n=10,
                        dpi=cfg.output.figure_dpi,
                    )

                    # Generate HTML report
                    html_path = dirs["results"] / "report.html"
                    generate_html_report(
                        scores_df=ranking_df,
                        summary_stats=summary_stats,
                        figures=figures,
                        output_path=html_path,
                        config=cfg.model_dump(),
                    )
                except Exception as e:
                    logger.warning(f"Visualization failed: {e}")
                    errors.append({
                        "error_type": "VisualizationError",
                        "message": str(e),
                    })

        memory.checkpoint("after_visualization")

        # Print summary
        timer.print_summary()
        memory.stop()

        click.echo("\n=== Analysis Complete ===")
        click.echo(f"Total sensors analyzed: {len(scores_df)}")
        click.echo(f"Confidence distribution (strict mode):")
        for level, count in summary_stats.get("confidence_distribution_strict", {}).items():
            click.echo(f"  {level}: {count}")
        click.echo(f"\nResults saved to: {dirs['results']}")
        click.echo(f"Figures saved to: {dirs['figures']}")
        click.echo(f"Logs saved to: {log_file}")

        # Save debug bundle
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
        logger.exception(f"Pipeline failed: {e}")
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
        click.echo(f"Error: {e}", err=True)
        click.echo(f"Debug bundle saved to: {bundle_path}", err=True)
        sys.exit(1)

    finally:
        shutdown_logger(sinks)
        gc.collect()


@main.command()
@click.option(
    "--log",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to log file",
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=Path("outputs/debug_bundles"),
    help="Output directory for debug bundle",
)
def debug(log: Path, output: Path):
    """Generate debug bundle from log file."""
    from seasonality.debug.bundler import DebugBundleBuilder, save_debug_bundle

    click.echo(f"Generating debug bundle from: {log}")

    builder = DebugBundleBuilder()

    # Read log file
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

    # Try to extract errors from log
    import re
    error_pattern = r"ERROR.*?:(.*?)(?=\d{4}-\d{2}-\d{2}|$)"
    errors = re.findall(error_pattern, log_content, re.DOTALL)
    for error in errors[:10]:  # Limit to 10 errors
        builder.add_error("LogError", error.strip()[:500])

    bundle = builder.build()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    bundle_path = output / f"bundle_{bundle.bundle_id}.json"
    save_debug_bundle(bundle, bundle_path, anonymize=True)

    click.echo(f"Debug bundle saved to: {bundle_path}")


@main.command()
@click.option(
    "--bundle", "-b",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to debug bundle JSON file",
)
def diagnose(bundle: Path):
    """Run LLM-based diagnostics on debug bundle."""
    from seasonality.debug.bundler import load_debug_bundle
    from seasonality.debug.llm_client import create_diagnostic_client

    click.echo(f"Loading debug bundle: {bundle}")
    debug_bundle = load_debug_bundle(bundle)

    click.echo("Running diagnostic analysis...")
    client = create_diagnostic_client()

    if not client.is_available:
        click.echo("Warning: LLM provider not configured. Set OPENAI_API_KEY or Azure OpenAI credentials.", err=True)
        click.echo("\nBasic analysis (without LLM):")
        click.echo(f"  Bundle ID: {debug_bundle.bundle_id}")
        click.echo(f"  Created: {debug_bundle.created_at}")
        click.echo(f"  Errors: {len(debug_bundle.errors)}")
        click.echo(f"  Warnings: {len(debug_bundle.warnings)}")

        if debug_bundle.errors:
            click.echo("\nErrors found:")
            for err in debug_bundle.errors[:5]:
                click.echo(f"  - [{err.get('type', 'Unknown')}] {err.get('message', '')[:100]}")
        return

    result = client.diagnose(debug_bundle)

    click.echo(f"\n=== Diagnostic Results ===")
    click.echo(f"Confidence: {result.confidence}")
    click.echo(f"\nSummary:\n{result.summary}")

    if result.issues_found:
        click.echo(f"\nIssues Found ({len(result.issues_found)}):")
        for issue in result.issues_found:
            severity = issue.get("severity", "unknown")
            category = issue.get("category", "unknown")
            desc = issue.get("description", "No description")
            click.echo(f"  [{severity.upper()}] ({category}) {desc}")

    if result.recommendations:
        click.echo(f"\nRecommendations:")
        for rec in result.recommendations:
            click.echo(f"  - {rec}")


@main.command()
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=Path("config"),
    help="Output directory for config files",
)
def init(output: Path):
    """Initialize configuration files."""
    from seasonality.config import SeasonalityConfig

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    # Create default config
    default_config = SeasonalityConfig()
    default_path = output / "default.yaml"
    default_config.to_yaml(default_path)
    click.echo(f"Created: {default_path}")

    # Create strict mode config
    from seasonality.config import ScoringConfig
    strict_config = SeasonalityConfig(
        scoring=ScoringConfig(mode="strict")
    )
    strict_path = output / "strict_mode.yaml"
    strict_config.to_yaml(strict_path)
    click.echo(f"Created: {strict_path}")

    # Create exploratory mode config
    exploratory_config = SeasonalityConfig(
        scoring=ScoringConfig(mode="exploratory")
    )
    exploratory_path = output / "exploratory_mode.yaml"
    exploratory_config.to_yaml(exploratory_path)
    click.echo(f"Created: {exploratory_path}")

    click.echo(f"\nConfiguration files created in: {output}")


if __name__ == "__main__":
    main()
