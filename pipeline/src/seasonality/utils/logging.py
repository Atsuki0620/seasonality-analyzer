"""Logging utilities for Seasonality Analyzer."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

# Track sink IDs so we can close file handles explicitly (important on Windows).


def setup_logger(
    log_file: Optional[Path] = None,
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "7 days",
    console: bool = True,
) -> dict[str, int]:
    """Configure loguru logger.

    Args:
        log_file: Path to log file. If None, file logging is disabled.
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        rotation: Log rotation setting.
        retention: Log retention setting.
        console: Whether to enable console logging.
    """
    # Remove default handler
    logger.remove()

    sink_ids: dict[str, int] = {}

    # Console handler
    if console:
        sink_ids["console"] = logger.add(
            sys.stderr,
            level=level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True,
        )

    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        sink_ids["file"] = logger.add(
            log_file,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation=rotation,
            retention=retention,
            encoding="utf-8",
        )

    return sink_ids


def get_logger(name: str = None):
    """Get logger instance.

    Args:
        name: Logger name (for context).

    Returns:
        Logger instance.
    """
    if name:
        return logger.bind(name=name)
    return logger


def shutdown_logger(sink_ids: dict[str, int] | None = None) -> None:
    """Remove sinks to close file handles.

    Loguru keeps file handles open until a sink is removed. On Windows this
    prevents TemporaryDirectory from deleting the log file at teardown, so we
    explicitly remove the sinks that were registered by :func:`setup_logger`.

    Args:
        sink_ids: Dictionary returned by ``setup_logger``.
    """

    if not sink_ids:
        return

    for sink_id in sink_ids.values():
        try:
            logger.remove(sink_id)
        except Exception:
            # Fail-safe: ignore errors during shutdown
            continue


class LogContext:
    """Context manager for adding context to log messages."""

    def __init__(self, **context):
        """Initialize with context key-value pairs."""
        self.context = context
        self._token = None

    def __enter__(self):
        """Enter context."""
        self._token = logger.contextualize(**self.context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        if self._token:
            self._token.__exit__(exc_type, exc_val, exc_tb)
        return False
