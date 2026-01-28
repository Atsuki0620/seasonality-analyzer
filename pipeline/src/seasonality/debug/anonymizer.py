"""Anonymization utilities for debug bundles."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

from seasonality.debug.bundler import DebugBundle


def anonymize_sensor_names(
    names: List[str],
    prefix: str = "SENSOR_",
) -> Dict[str, str]:
    """Create mapping from original sensor names to anonymized names.

    Args:
        names: List of original sensor names.
        prefix: Prefix for anonymized names.

    Returns:
        Dictionary mapping original names to anonymized names.
    """
    mapping = {}
    for i, name in enumerate(sorted(set(names)), start=1):
        mapping[name] = f"{prefix}{i:03d}"
    return mapping


def mask_paths(
    text: str,
    patterns: List[str] = None,
) -> str:
    """Mask file paths in text.

    Args:
        text: Input text.
        patterns: Additional patterns to mask.

    Returns:
        Text with paths masked.
    """
    # Common path patterns
    default_patterns = [
        r"[A-Za-z]:\\[^\s\"']+",  # Windows paths
        r"/[^\s\"']+(?:/[^\s\"']+)+",  # Unix paths
        r"~[^\s\"']+",  # Home directory paths
    ]

    patterns = patterns or default_patterns

    result = text
    for pattern in patterns:
        result = re.sub(pattern, "[PATH_MASKED]", result)

    return result


def anonymize_dict(
    data: Dict[str, Any],
    sensor_mapping: Dict[str, str],
    mask_path_values: bool = True,
) -> Dict[str, Any]:
    """Recursively anonymize dictionary.

    Args:
        data: Dictionary to anonymize.
        sensor_mapping: Sensor name mapping.
        mask_path_values: Whether to mask path values.

    Returns:
        Anonymized dictionary.
    """
    result = {}

    for key, value in data.items():
        # Check if key is a sensor name
        new_key = sensor_mapping.get(key, key)

        if isinstance(value, dict):
            result[new_key] = anonymize_dict(value, sensor_mapping, mask_path_values)
        elif isinstance(value, list):
            result[new_key] = anonymize_list(value, sensor_mapping, mask_path_values)
        elif isinstance(value, str):
            # Replace sensor names in string values
            new_value = value
            for original, anonymized in sensor_mapping.items():
                new_value = new_value.replace(original, anonymized)

            # Mask paths if needed
            if mask_path_values:
                new_value = mask_paths(new_value)

            result[new_key] = new_value
        else:
            result[new_key] = value

    return result


def anonymize_list(
    data: List[Any],
    sensor_mapping: Dict[str, str],
    mask_path_values: bool = True,
) -> List[Any]:
    """Recursively anonymize list.

    Args:
        data: List to anonymize.
        sensor_mapping: Sensor name mapping.
        mask_path_values: Whether to mask path values.

    Returns:
        Anonymized list.
    """
    result = []

    for item in data:
        if isinstance(item, dict):
            result.append(anonymize_dict(item, sensor_mapping, mask_path_values))
        elif isinstance(item, list):
            result.append(anonymize_list(item, sensor_mapping, mask_path_values))
        elif isinstance(item, str):
            new_value = item
            for original, anonymized in sensor_mapping.items():
                new_value = new_value.replace(original, anonymized)
            if mask_path_values:
                new_value = mask_paths(new_value)
            result.append(new_value)
        else:
            result.append(item)

    return result


def anonymize_bundle(
    bundle: DebugBundle,
    sensor_prefix: str = "SENSOR_",
    mask_paths_enabled: bool = True,
) -> DebugBundle:
    """Anonymize a debug bundle.

    Args:
        bundle: Debug bundle to anonymize.
        sensor_prefix: Prefix for anonymized sensor names.
        mask_paths_enabled: Whether to mask file paths.

    Returns:
        Anonymized debug bundle.
    """
    # Get sensor names from data summary
    sensor_names = bundle.data_summary.get("sensor_columns", [])
    sensor_mapping = anonymize_sensor_names(sensor_names, sensor_prefix)

    # Anonymize config
    anon_config = anonymize_dict(bundle.config, sensor_mapping, mask_paths_enabled)

    # Anonymize data summary
    anon_data_summary = anonymize_dict(bundle.data_summary, sensor_mapping, mask_paths_enabled)

    # Update sensor columns list
    if "sensor_columns" in anon_data_summary:
        anon_data_summary["sensor_columns"] = [
            sensor_mapping.get(s, s) for s in bundle.data_summary.get("sensor_columns", [])
        ]

    # Anonymize health checks
    anon_health_checks = []
    for hc in bundle.health_checks:
        anon_hc = type(hc)(
            check_name=hc.check_name,
            passed=hc.passed,
            message=_anonymize_string(hc.message, sensor_mapping, mask_paths_enabled),
            details=anonymize_dict(hc.details, sensor_mapping, mask_paths_enabled),
        )
        anon_health_checks.append(anon_hc)

    # Anonymize processing steps
    anon_steps = []
    for step in bundle.processing_steps:
        anon_step = type(step)(
            step_name=step.step_name,
            start_time=step.start_time,
            end_time=step.end_time,
            duration_seconds=step.duration_seconds,
            status=step.status,
            message=_anonymize_string(step.message, sensor_mapping, mask_paths_enabled) if step.message else None,
            metrics=anonymize_dict(step.metrics, sensor_mapping, mask_paths_enabled),
        )
        anon_steps.append(anon_step)

    # Anonymize errors
    anon_errors = anonymize_list(bundle.errors, sensor_mapping, mask_paths_enabled)

    # Anonymize warnings
    anon_warnings = [_anonymize_string(w, sensor_mapping, mask_paths_enabled) for w in bundle.warnings]

    # Anonymize results summary
    anon_results = anonymize_dict(bundle.results_summary, sensor_mapping, mask_paths_enabled) if bundle.results_summary else None

    # Anonymize log excerpt
    anon_log = _anonymize_string(bundle.log_excerpt, sensor_mapping, mask_paths_enabled) if bundle.log_excerpt else None

    return DebugBundle(
        bundle_id=bundle.bundle_id,
        created_at=bundle.created_at,
        version=bundle.version,
        config=anon_config,
        data_summary=anon_data_summary,
        health_checks=anon_health_checks,
        processing_steps=anon_steps,
        errors=anon_errors,
        warnings=anon_warnings,
        results_summary=anon_results,
        log_excerpt=anon_log,
    )


def _anonymize_string(
    text: str,
    sensor_mapping: Dict[str, str],
    mask_paths_enabled: bool,
) -> str:
    """Anonymize a string value."""
    if not text:
        return text

    result = text
    for original, anonymized in sensor_mapping.items():
        result = result.replace(original, anonymized)

    if mask_paths_enabled:
        result = mask_paths(result)

    return result
