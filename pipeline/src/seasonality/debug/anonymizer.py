"""デバッグバンドルの匿名化ユーティリティ"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

from seasonality.debug.bundler import DebugBundle


def anonymize_sensor_names(
    names: List[str],
    prefix: str = "SENSOR_",
) -> Dict[str, str]:
    """元のセンサー名から匿名化された名前へのマッピングを作成

    Args:
        names: 元のセンサー名のリスト
        prefix: 匿名化された名前のプレフィックス

    Returns:
        元の名前から匿名化された名前へのマッピング辞書
    """
    mapping = {}
    for i, name in enumerate(sorted(set(names)), start=1):
        mapping[name] = f"{prefix}{i:03d}"
    return mapping


def mask_paths(
    text: str,
    patterns: List[str] = None,
) -> str:
    """テキスト内のファイルパスをマスク

    Args:
        text: 入力テキスト
        patterns: マスクする追加のパターン

    Returns:
        パスがマスクされたテキスト
    """
    # 一般的なパスパターン
    default_patterns = [
        r"[A-Za-z]:\\[^\s\"']+",  # Windowsパス
        r"/[^\s\"']+(?:/[^\s\"']+)+",  # Unixパス
        r"~[^\s\"']+",  # ホームディレクトリパス
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
    """辞書を再帰的に匿名化

    Args:
        data: 匿名化する辞書
        sensor_mapping: センサー名のマッピング
        mask_path_values: パス値をマスクするかどうか

    Returns:
        匿名化された辞書
    """
    result = {}

    for key, value in data.items():
        # キーがセンサー名かどうかをチェック
        new_key = sensor_mapping.get(key, key)

        if isinstance(value, dict):
            result[new_key] = anonymize_dict(value, sensor_mapping, mask_path_values)
        elif isinstance(value, list):
            result[new_key] = anonymize_list(value, sensor_mapping, mask_path_values)
        elif isinstance(value, str):
            # 文字列値内のセンサー名を置換
            new_value = value
            for original, anonymized in sensor_mapping.items():
                new_value = new_value.replace(original, anonymized)

            # 必要に応じてパスをマスク
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
    """リストを再帰的に匿名化

    Args:
        data: 匿名化するリスト
        sensor_mapping: センサー名のマッピング
        mask_path_values: パス値をマスクするかどうか

    Returns:
        匿名化されたリスト
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
    """デバッグバンドルを匿名化

    Args:
        bundle: 匿名化するデバッグバンドル
        sensor_prefix: 匿名化されたセンサー名のプレフィックス
        mask_paths_enabled: ファイルパスをマスクするかどうか

    Returns:
        匿名化されたデバッグバンドル
    """
    # データサマリーからセンサー名を取得
    sensor_names = bundle.data_summary.get("sensor_columns", [])
    sensor_mapping = anonymize_sensor_names(sensor_names, sensor_prefix)

    # 設定を匿名化
    anon_config = anonymize_dict(bundle.config, sensor_mapping, mask_paths_enabled)

    # データサマリーを匿名化
    anon_data_summary = anonymize_dict(bundle.data_summary, sensor_mapping, mask_paths_enabled)

    # センサーカラムリストを更新
    if "sensor_columns" in anon_data_summary:
        anon_data_summary["sensor_columns"] = [
            sensor_mapping.get(s, s) for s in bundle.data_summary.get("sensor_columns", [])
        ]

    # ヘルスチェックを匿名化
    anon_health_checks = []
    for hc in bundle.health_checks:
        anon_hc = type(hc)(
            check_name=hc.check_name,
            passed=hc.passed,
            message=_anonymize_string(hc.message, sensor_mapping, mask_paths_enabled),
            details=anonymize_dict(hc.details, sensor_mapping, mask_paths_enabled),
        )
        anon_health_checks.append(anon_hc)

    # 処理ステップを匿名化
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

    # エラーを匿名化
    anon_errors = anonymize_list(bundle.errors, sensor_mapping, mask_paths_enabled)

    # 警告を匿名化
    anon_warnings = [_anonymize_string(w, sensor_mapping, mask_paths_enabled) for w in bundle.warnings]

    # 結果サマリーを匿名化
    anon_results = anonymize_dict(bundle.results_summary, sensor_mapping, mask_paths_enabled) if bundle.results_summary else None

    # ログ抜粋を匿名化
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
    """文字列値を匿名化"""
    if not text:
        return text

    result = text
    for original, anonymized in sensor_mapping.items():
        result = result.replace(original, anonymized)

    if mask_paths_enabled:
        result = mask_paths(result)

    return result
