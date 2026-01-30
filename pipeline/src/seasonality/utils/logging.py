"""季節性分析のためのロギングユーティリティ。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

# シンクIDを追跡してファイルハンドルを明示的に閉じる（Windows では重要）。


def setup_logger(
    log_file: Optional[Path] = None,
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "7 days",
    console: bool = True,
) -> dict[str, int]:
    """loguru ロガーを設定する。

    Args:
        log_file: ログファイルへのパス。Noneの場合、ファイルログは無効化される。
        level: ログレベル（DEBUG、INFO、WARNING、ERROR）。
        rotation: ログローテーション設定。
        retention: ログ保持設定。
        console: コンソールログを有効化するかどうか。
    """
    # デフォルトハンドラを削除
    logger.remove()

    sink_ids: dict[str, int] = {}

    # コンソールハンドラ
    if console:
        sink_ids["console"] = logger.add(
            sys.stderr,
            level=level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True,
        )

    # ファイルハンドラ
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
    """ロガーインスタンスを取得する。

    Args:
        name: ロガー名（コンテキスト用）。

    Returns:
        ロガーインスタンス。
    """
    if name:
        return logger.bind(name=name)
    return logger


def shutdown_logger(sink_ids: dict[str, int] | None = None) -> None:
    """シンクを削除してファイルハンドルを閉じる。

    Loguruはシンクが削除されるまでファイルハンドルを開いたままにする。Windowsでは
    これによりTemporaryDirectoryがクリーンアップ時にログファイルを削除できないため、
    :func:`setup_logger`によって登録されたシンクを明示的に削除する。

    Args:
        sink_ids: ``setup_logger``が返す辞書。
    """

    if not sink_ids:
        return

    for sink_id in sink_ids.values():
        try:
            logger.remove(sink_id)
        except Exception:
            # フェイルセーフ：シャットダウン中のエラーは無視
            continue


class LogContext:
    """ログメッセージにコンテキストを追加するためのコンテキストマネージャ。"""

    def __init__(self, **context):
        """コンテキストのキーと値のペアで初期化する。"""
        self.context = context
        self._token = None

    def __enter__(self):
        """コンテキストに入る。"""
        self._token = logger.contextualize(**self.context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストから出る。"""
        if self._token:
            self._token.__exit__(exc_type, exc_val, exc_tb)
        return False
