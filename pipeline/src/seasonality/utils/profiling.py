"""Profiling utilities for Seasonality Analyzer."""

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
    """Memory usage snapshot."""

    timestamp: datetime
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    percent: float

    @classmethod
    def capture(cls) -> "MemorySnapshot":
        """Capture current memory usage."""
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
    """Track memory usage during processing."""

    def __init__(self, name: str = "default"):
        """Initialize tracker.

        Args:
            name: Tracker name for logging.
        """
        self.name = name
        self.snapshots: List[MemorySnapshot] = []
        self._start_snapshot: Optional[MemorySnapshot] = None

    def start(self) -> "MemoryTracker":
        """Start tracking."""
        gc.collect()
        self._start_snapshot = MemorySnapshot.capture()
        self.snapshots = [self._start_snapshot]
        return self

    def checkpoint(self, label: str = "") -> MemorySnapshot:
        """Take a memory checkpoint.

        Args:
            label: Optional label for the checkpoint.

        Returns:
            Current memory snapshot.
        """
        snapshot = MemorySnapshot.capture()
        self.snapshots.append(snapshot)

        if self._start_snapshot:
            delta = snapshot.rss_mb - self._start_snapshot.rss_mb
            logger.debug(f"[{self.name}] Memory checkpoint {label}: {snapshot.rss_mb:.1f}MB (delta: {delta:+.1f}MB)")

        return snapshot

    def stop(self) -> Dict[str, Any]:
        """Stop tracking and return summary.

        Returns:
            Dictionary with memory usage summary.
        """
        gc.collect()
        final_snapshot = MemorySnapshot.capture()
        self.snapshots.append(final_snapshot)

        if not self._start_snapshot:
            return {"error": "Tracker not started"}

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
            f"[{self.name}] Memory: {summary['start_mb']:.1f}MB -> {summary['end_mb']:.1f}MB "
            f"(delta: {delta:+.1f}MB, peak: {summary['peak_mb']:.1f}MB)"
        )

        return summary


@dataclass
class TimingResult:
    """Result of timing measurement."""

    name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float


class StepTimer:
    """Timer for tracking processing steps."""

    def __init__(self):
        """Initialize timer."""
        self.steps: List[TimingResult] = []
        self._current_step: Optional[str] = None
        self._step_start: Optional[datetime] = None

    def start_step(self, name: str) -> "StepTimer":
        """Start timing a step.

        Args:
            name: Step name.

        Returns:
            Self for chaining.
        """
        if self._current_step:
            self.end_step()

        self._current_step = name
        self._step_start = datetime.now()
        logger.debug(f"Started step: {name}")
        return self

    def end_step(self) -> Optional[TimingResult]:
        """End current step.

        Returns:
            Timing result for the step.
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
        logger.debug(f"Completed step '{self._current_step}': {duration:.2f}s")

        self._current_step = None
        self._step_start = None

        return result

    @contextmanager
    def step(self, name: str):
        """Context manager for timing a step.

        Args:
            name: Step name.
        """
        self.start_step(name)
        try:
            yield
        finally:
            self.end_step()

    def summary(self) -> Dict[str, Any]:
        """Get timing summary.

        Returns:
            Dictionary with timing summary.
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
        """Print timing summary to logger."""
        summary = self.summary()
        logger.info(f"Total time: {summary['total_seconds']:.2f}s")
        for step in summary["steps"]:
            logger.info(f"  {step['name']}: {step['duration_seconds']:.2f}s ({step['percent']:.1f}%)")


def time_it(func: Callable = None, *, name: str = None):
    """Decorator to time function execution.

    Args:
        func: Function to decorate.
        name: Optional name for logging.

    Returns:
        Decorated function.
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            func_name = name or f.__name__
            start = time.perf_counter()
            try:
                result = f(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.debug(f"{func_name} completed in {elapsed:.2f}s")
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                logger.error(f"{func_name} failed after {elapsed:.2f}s: {e}")
                raise
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator
