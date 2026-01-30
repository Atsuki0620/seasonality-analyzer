"""季節性分析のためのプロファイリングユーティリティ。"""

from __future__ import annotations

import functools
import gc
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class MemorySnapshot:
    """メモリ使用量のスナップショット。"""

    timestamp: datetime
    rss_mb: float  # 常駐セットサイズ
    vms_mb: float  # 仮想メモリサイズ
    percent: float

    @classmethod
    def capture(cls) -> "MemorySnapshot":
        """現在のメモリ使用量をキャプチャする。"""
        if not PSUTIL_AVAILABLE:
            return cls(
                timestamp=datetime.now(),
                rss_mb=0.0,
                vms_mb=0.0,
                percent=0.0,
            )

        process = psutil.Process()
        mem_info = process.memory_info()

        return cls(
            timestamp=datetime.now(),
            rss_mb=mem_info.rss / 1024 / 1024,
            vms_mb=mem_info.vms / 1024 / 1024,
            percent=process.memory_percent(),
        )


class MemoryTracker:
    """処理中のメモリ使用量を追跡する。"""

    def __init__(self, name: str = "default"):
        """トラッカーを初期化する。

        Args:
            name: ログ用のトラッカー名。
        """
        self.name = name
        self.snapshots: List[MemorySnapshot] = []
        self._start_snapshot: Optional[MemorySnapshot] = None

    def start(self) -> "MemoryTracker":
        """追跡を開始する。"""
        gc.collect()
        self._start_snapshot = MemorySnapshot.capture()
        self.snapshots = [self._start_snapshot]
        return self

    def checkpoint(self, label: str = "") -> MemorySnapshot:
        """メモリチェックポイントを作成する。

        Args:
            label: チェックポイントのオプションラベル。

        Returns:
            現在のメモリスナップショット。
        """
        snapshot = MemorySnapshot.capture()
        self.snapshots.append(snapshot)

        if self._start_snapshot:
            delta = snapshot.rss_mb - self._start_snapshot.rss_mb
            logger.debug(f"[{self.name}] メモリチェックポイント {label}: {snapshot.rss_mb:.1f}MB (差分: {delta:+.1f}MB)")

        return snapshot

    def stop(self) -> Dict[str, Any]:
        """追跡を停止してサマリーを返す。

        Returns:
            メモリ使用量サマリーを含む辞書。
        """
        gc.collect()
        final_snapshot = MemorySnapshot.capture()
        self.snapshots.append(final_snapshot)

        if not self._start_snapshot:
            return {"error": "トラッカーが開始されていません"}

        delta = final_snapshot.rss_mb - self._start_snapshot.rss_mb

        summary = {
            "name": self.name,
            "start_mb": self._start_snapshot.rss_mb,
            "end_mb": final_snapshot.rss_mb,
            "delta_mb": delta,
            "peak_mb": max(s.rss_mb for s in self.snapshots),
            "num_checkpoints": len(self.snapshots),
        }

        logger.info(
            f"[{self.name}] メモリ: {summary['start_mb']:.1f}MB -> {summary['end_mb']:.1f}MB "
            f"(差分: {delta:+.1f}MB, ピーク: {summary['peak_mb']:.1f}MB)"
        )

        return summary


@dataclass
class TimingResult:
    """タイミング測定の結果。"""

    name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float


class StepTimer:
    """処理ステップを追跡するためのタイマー。"""

    def __init__(self):
        """タイマーを初期化する。"""
        self.steps: List[TimingResult] = []
        self._current_step: Optional[str] = None
        self._step_start: Optional[datetime] = None

    def start_step(self, name: str) -> "StepTimer":
        """ステップのタイミング測定を開始する。

        Args:
            name: ステップ名。

        Returns:
            チェーン用の自身。
        """
        if self._current_step:
            self.end_step()

        self._current_step = name
        self._step_start = datetime.now()
        logger.debug(f"ステップを開始: {name}")
        return self

    def end_step(self) -> Optional[TimingResult]:
        """現在のステップを終了する。

        Returns:
            ステップのタイミング結果。
        """
        if not self._current_step or not self._step_start:
            return None

        end_time = datetime.now()
        duration = (end_time - self._step_start).total_seconds()

        result = TimingResult(
            name=self._current_step,
            start_time=self._step_start,
            end_time=end_time,
            duration_seconds=duration,
        )

        self.steps.append(result)
        logger.debug(f"ステップ '{self._current_step}' を完了: {duration:.2f}秒")

        self._current_step = None
        self._step_start = None

        return result

    @contextmanager
    def step(self, name: str):
        """ステップのタイミング測定用コンテキストマネージャ。

        Args:
            name: ステップ名。
        """
        self.start_step(name)
        try:
            yield
        finally:
            self.end_step()

    def summary(self) -> Dict[str, Any]:
        """タイミングサマリーを取得する。

        Returns:
            タイミングサマリーを含む辞書。
        """
        if not self.steps:
            return {"total_seconds": 0, "steps": []}

        total = sum(s.duration_seconds for s in self.steps)

        return {
            "total_seconds": total,
            "steps": [
                {
                    "name": s.name,
                    "duration_seconds": s.duration_seconds,
                    "percent": (s.duration_seconds / total * 100) if total > 0 else 0,
                }
                for s in self.steps
            ],
        }

    def print_summary(self) -> None:
        """タイミングサマリーをロガーに出力する。"""
        summary = self.summary()
        logger.info(f"合計時間: {summary['total_seconds']:.2f}秒")
        for step in summary["steps"]:
            logger.info(f"  {step['name']}: {step['duration_seconds']:.2f}秒 ({step['percent']:.1f}%)")


def time_it(func: Callable = None, *, name: str = None):
    """関数実行時間を計測するデコレータ。

    Args:
        func: デコレートする関数。
        name: ログ用のオプション名。

    Returns:
        デコレートされた関数。
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            func_name = name or f.__name__
            start = time.perf_counter()
            try:
                result = f(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.debug(f"{func_name} が {elapsed:.2f}秒 で完了")
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                logger.error(f"{func_name} が {elapsed:.2f}秒 後に失敗: {e}")
                raise
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator
