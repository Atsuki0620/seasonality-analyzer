"""デバッグバンドルの作成と管理"""

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
    """ヘルスチェックの結果"""

    check_name: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingStep:
    """処理ステップの記録"""

    step_name: str
    start_time: str
    end_time: str
    duration_seconds: float
    status: str  # 'success', 'warning', 'error'
    message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DebugBundle:
    """すべての診断情報を含むデバッグバンドル"""

    bundle_id: str
    created_at: str
    version: str

    # 設定
    config: Dict[str, Any]

    # データサマリー
    data_summary: Dict[str, Any]

    # ヘルスチェック
    health_checks: List[HealthCheckResult]

    # 処理ステップ
    processing_steps: List[ProcessingStep]

    # エラーと警告
    errors: List[Dict[str, Any]]
    warnings: List[str]

    # 結果サマリー
    results_summary: Optional[Dict[str, Any]] = None

    # ログ抜粋
    log_excerpt: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
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
        """辞書から作成"""
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
    """デバッグバンドルを作成するためのビルダー"""

    def __init__(self, version: str = "0.1.0"):
        """ビルダーを初期化"""
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
        """設定をセット"""
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
        """データサマリーをセット"""
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
        """ヘルスチェック結果を追加"""
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
        """処理ステップを追加"""
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
        """エラーを追加"""
        self.errors.append({
            "type": error_type,
            "message": message,
            "traceback": traceback,
            "context": context or {},
        })
        return self

    def add_warning(self, message: str) -> "DebugBundleBuilder":
        """警告を追加"""
        self.warnings.append(message)
        return self

    def set_results_summary(self, summary: Dict[str, Any]) -> "DebugBundleBuilder":
        """結果サマリーをセット"""
        self.results_summary = summary
        return self

    def set_log_excerpt(self, log_text: str, max_lines: int = 100) -> "DebugBundleBuilder":
        """ログ抜粋をセット"""
        lines = log_text.split("\n")
        if len(lines) > max_lines:
            lines = lines[-max_lines:]
        self.log_excerpt = "\n".join(lines)
        return self

    def build(self) -> DebugBundle:
        """デバッグバンドルを構築"""
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
    """分析結果からデバッグバンドルを作成

    Args:
        config: 設定辞書
        data_health: DataHealthReportオブジェクト
        processing_log: 処理ステップ記録のリスト
        results_summary: 分析結果サマリー
        errors: 発生したエラーのリスト
        log_file: ログ抜粋用のログファイルパス

    Returns:
        DebugBundleインスタンス
    """
    builder = DebugBundleBuilder()

    # 設定をセット
    builder.set_config(config)

    # ヘルスレポートからデータサマリーをセット
    if hasattr(data_health, "total_records"):
        builder.set_data_summary(
            total_records=data_health.total_records,
            date_range=data_health.date_range,
            sensor_count=data_health.sensor_count,
            missing_rate=data_health.missing_rate,
            sensor_columns=data_health.sensor_columns,
        )

        # ヘルスチェックを追加
        builder.add_health_check(
            "data_loaded",
            data_health.total_records > 0,
            f"{data_health.total_records}件のレコードを読み込みました",
        )
        builder.add_health_check(
            "missing_rate",
            data_health.missing_rate < 0.1,
            f"欠損率: {data_health.missing_rate:.2%}",
        )

        for warning in data_health.warnings:
            builder.add_warning(warning)

    # 処理ステップを追加
    if processing_log:
        for step in processing_log:
            builder.add_processing_step(**step)

    # 結果サマリーをセット
    if results_summary:
        builder.set_results_summary(results_summary)

    # エラーを追加
    if errors:
        for error in errors:
            builder.add_error(**error)

    # ログ抜粋を追加
    if log_file and log_file.exists():
        with open(log_file, "r", encoding="utf-8") as f:
            builder.set_log_excerpt(f.read())

    return builder.build()


def save_debug_bundle(
    bundle: DebugBundle,
    output_path: Path,
    anonymize: bool = True,
) -> Path:
    """デバッグバンドルをJSONファイルに保存

    Args:
        bundle: 保存するデバッグバンドル
        output_path: 出力ファイルパス
        anonymize: センサー名を匿名化するかどうか

    Returns:
        保存されたファイルのパス
    """
    from seasonality.debug.anonymizer import anonymize_bundle

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if anonymize:
        bundle = anonymize_bundle(bundle)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(bundle.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info(f"デバッグバンドルを保存しました: {output_path}")
    return output_path


def load_debug_bundle(path: Path) -> DebugBundle:
    """JSONファイルからデバッグバンドルを読み込み

    Args:
        path: バンドルファイルのパス

    Returns:
        DebugBundleインスタンス
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return DebugBundle.from_dict(data)
