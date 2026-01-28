"""Debug bundle creation and management."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    check_name: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingStep:
    """Record of a processing step."""

    step_name: str
    start_time: str
    end_time: str
    duration_seconds: float
    status: str  # 'success', 'warning', 'error'
    message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DebugBundle:
    """Debug bundle containing all diagnostic information."""

    bundle_id: str
    created_at: str
    version: str

    # Configuration
    config: Dict[str, Any]

    # Data summary
    data_summary: Dict[str, Any]

    # Health checks
    health_checks: List[HealthCheckResult]

    # Processing steps
    processing_steps: List[ProcessingStep]

    # Errors and warnings
    errors: List[Dict[str, Any]]
    warnings: List[str]

    # Results summary
    results_summary: Optional[Dict[str, Any]] = None

    # Log excerpt
    log_excerpt: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bundle_id": self.bundle_id,
            "created_at": self.created_at,
            "version": self.version,
            "config": self.config,
            "data_summary": self.data_summary,
            "health_checks": [asdict(hc) for hc in self.health_checks],
            "processing_steps": [asdict(ps) for ps in self.processing_steps],
            "errors": self.errors,
            "warnings": self.warnings,
            "results_summary": self.results_summary,
            "log_excerpt": self.log_excerpt,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DebugBundle":
        """Create from dictionary."""
        return cls(
            bundle_id=data["bundle_id"],
            created_at=data["created_at"],
            version=data["version"],
            config=data["config"],
            data_summary=data["data_summary"],
            health_checks=[HealthCheckResult(**hc) for hc in data["health_checks"]],
            processing_steps=[ProcessingStep(**ps) for ps in data["processing_steps"]],
            errors=data["errors"],
            warnings=data["warnings"],
            results_summary=data.get("results_summary"),
            log_excerpt=data.get("log_excerpt"),
        )


class DebugBundleBuilder:
    """Builder for creating debug bundles."""

    def __init__(self, version: str = "0.1.0"):
        """Initialize builder."""
        self.version = version
        self.bundle_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.config: Dict[str, Any] = {}
        self.data_summary: Dict[str, Any] = {}
        self.health_checks: List[HealthCheckResult] = []
        self.processing_steps: List[ProcessingStep] = []
        self.errors: List[Dict[str, Any]] = []
        self.warnings: List[str] = []
        self.results_summary: Optional[Dict[str, Any]] = None
        self.log_excerpt: Optional[str] = None

    def set_config(self, config: Dict[str, Any]) -> "DebugBundleBuilder":
        """Set configuration."""
        self.config = config
        return self

    def set_data_summary(
        self,
        total_records: int,
        date_range: tuple,
        sensor_count: int,
        missing_rate: float,
        sensor_columns: Optional[List[str]] = None,
    ) -> "DebugBundleBuilder":
        """Set data summary."""
        self.data_summary = {
            "total_records": total_records,
            "date_range": {
                "start": str(date_range[0]),
                "end": str(date_range[1]),
            },
            "sensor_count": sensor_count,
            "missing_rate": missing_rate,
            "sensor_columns": sensor_columns or [],
        }
        return self

    def add_health_check(
        self,
        name: str,
        passed: bool,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> "DebugBundleBuilder":
        """Add health check result."""
        self.health_checks.append(HealthCheckResult(
            check_name=name,
            passed=passed,
            message=message,
            details=details or {},
        ))
        return self

    def add_processing_step(
        self,
        name: str,
        start_time: datetime,
        end_time: datetime,
        status: str,
        message: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> "DebugBundleBuilder":
        """Add processing step."""
        duration = (end_time - start_time).total_seconds()
        self.processing_steps.append(ProcessingStep(
            step_name=name,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=duration,
            status=status,
            message=message,
            metrics=metrics or {},
        ))
        return self

    def add_error(
        self,
        error_type: str,
        message: str,
        traceback: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> "DebugBundleBuilder":
        """Add error."""
        self.errors.append({
            "type": error_type,
            "message": message,
            "traceback": traceback,
            "context": context or {},
        })
        return self

    def add_warning(self, message: str) -> "DebugBundleBuilder":
        """Add warning."""
        self.warnings.append(message)
        return self

    def set_results_summary(self, summary: Dict[str, Any]) -> "DebugBundleBuilder":
        """Set results summary."""
        self.results_summary = summary
        return self

    def set_log_excerpt(self, log_text: str, max_lines: int = 100) -> "DebugBundleBuilder":
        """Set log excerpt."""
        lines = log_text.split("\n")
        if len(lines) > max_lines:
            lines = lines[-max_lines:]
        self.log_excerpt = "\n".join(lines)
        return self

    def build(self) -> DebugBundle:
        """Build the debug bundle."""
        return DebugBundle(
            bundle_id=self.bundle_id,
            created_at=datetime.now().isoformat(),
            version=self.version,
            config=self.config,
            data_summary=self.data_summary,
            health_checks=self.health_checks,
            processing_steps=self.processing_steps,
            errors=self.errors,
            warnings=self.warnings,
            results_summary=self.results_summary,
            log_excerpt=self.log_excerpt,
        )


def create_debug_bundle(
    config: Dict[str, Any],
    data_health: Any,
    processing_log: Optional[List[Dict[str, Any]]] = None,
    results_summary: Optional[Dict[str, Any]] = None,
    errors: Optional[List[Dict[str, Any]]] = None,
    log_file: Optional[Path] = None,
) -> DebugBundle:
    """Create a debug bundle from analysis results.

    Args:
        config: Configuration dictionary.
        data_health: DataHealthReport object.
        processing_log: List of processing step records.
        results_summary: Analysis results summary.
        errors: List of errors encountered.
        log_file: Path to log file for excerpt.

    Returns:
        DebugBundle instance.
    """
    builder = DebugBundleBuilder()

    # Set config
    builder.set_config(config)

    # Set data summary from health report
    if hasattr(data_health, "total_records"):
        builder.set_data_summary(
            total_records=data_health.total_records,
            date_range=data_health.date_range,
            sensor_count=data_health.sensor_count,
            missing_rate=data_health.missing_rate,
            sensor_columns=data_health.sensor_columns,
        )

        # Add health checks
        builder.add_health_check(
            "data_loaded",
            data_health.total_records > 0,
            f"Loaded {data_health.total_records} records",
        )
        builder.add_health_check(
            "missing_rate",
            data_health.missing_rate < 0.1,
            f"Missing rate: {data_health.missing_rate:.2%}",
        )

        for warning in data_health.warnings:
            builder.add_warning(warning)

    # Add processing steps
    if processing_log:
        for step in processing_log:
            builder.add_processing_step(**step)

    # Set results summary
    if results_summary:
        builder.set_results_summary(results_summary)

    # Add errors
    if errors:
        for error in errors:
            builder.add_error(**error)

    # Add log excerpt
    if log_file and log_file.exists():
        with open(log_file, "r", encoding="utf-8") as f:
            builder.set_log_excerpt(f.read())

    return builder.build()


def save_debug_bundle(
    bundle: DebugBundle,
    output_path: Path,
    anonymize: bool = True,
) -> Path:
    """Save debug bundle to JSON file.

    Args:
        bundle: Debug bundle to save.
        output_path: Output file path.
        anonymize: Whether to anonymize sensor names.

    Returns:
        Path to saved file.
    """
    from seasonality.debug.anonymizer import anonymize_bundle

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if anonymize:
        bundle = anonymize_bundle(bundle)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(bundle.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info(f"Saved debug bundle: {output_path}")
    return output_path


def load_debug_bundle(path: Path) -> DebugBundle:
    """Load debug bundle from JSON file.

    Args:
        path: Path to bundle file.

    Returns:
        DebugBundle instance.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return DebugBundle.from_dict(data)
