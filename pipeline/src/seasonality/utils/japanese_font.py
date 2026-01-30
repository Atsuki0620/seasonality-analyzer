"""日本語フォント設定ユーティリティ。

matplotlibとseabornで日本語を正しく表示するための設定を行います。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from loguru import logger


def setup_japanese_font() -> None:
    """matplotlibの日本語フォント設定を行う。

    利用可能な日本語フォントを自動検出して設定します。
    フォントが見つからない場合は警告を出力します。
    """
    # 日本語フォントの候補リスト（優先順）
    japanese_fonts = [
        'IPAexGothic',      # IPAフォント（Linux）
        'IPAGothic',        # IPAフォント（Linux）
        'Noto Sans CJK JP', # Google Notoフォント
        'Noto Sans JP',     # Google Notoフォント
        'Yu Gothic',        # 游ゴシック（Windows/Mac）
        'Meiryo',           # メイリオ（Windows）
        'Hiragino Sans',    # ヒラギノ（Mac）
        'Hiragino Kaku Gothic Pro',  # ヒラギノ角ゴ（Mac）
        'MS Gothic',        # MSゴシック（Windows）
        'TakaoPGothic',     # Takaoフォント（Linux）
    ]

    # システムで利用可能なフォントを取得
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    # 利用可能な日本語フォントを検索
    selected_font = None
    for font in japanese_fonts:
        if font in available_fonts:
            selected_font = font
            break

    if selected_font:
        # フォント設定を適用
        plt.rcParams['font.family'] = selected_font
        plt.rcParams['font.sans-serif'] = [selected_font] + plt.rcParams['font.sans-serif']

        # マイナス記号の文字化け対策
        plt.rcParams['axes.unicode_minus'] = False

        logger.info(f"日本語フォント '{selected_font}' を設定しました")
    else:
        # フォントが見つからない場合の警告
        logger.warning(
            "日本語フォントが見つかりませんでした。グラフの日本語が文字化けする可能性があります。"
        )
        logger.warning(
            "以下のいずれかのフォントをインストールすることを推奨します: "
            f"{', '.join(japanese_fonts[:3])}"
        )

        # フォールバック設定（マイナス記号のみ対策）
        plt.rcParams['axes.unicode_minus'] = False


def get_available_japanese_fonts() -> list[str]:
    """システムで利用可能な日本語フォントのリストを取得する。

    Returns:
        利用可能な日本語フォント名のリスト。
    """
    japanese_font_keywords = [
        'Gothic', 'ゴシック', 'Mincho', '明朝',
        'IPA', 'Noto', 'Yu', 'Meiryo', 'メイリオ',
        'Hiragino', 'ヒラギノ', 'Takao',
    ]

    available_fonts = [f.name for f in fm.fontManager.ttflist]

    japanese_fonts = []
    for font in available_fonts:
        if any(keyword in font for keyword in japanese_font_keywords):
            if font not in japanese_fonts:
                japanese_fonts.append(font)

    return sorted(japanese_fonts)


def print_available_fonts() -> None:
    """利用可能な日本語フォントを表示する（デバッグ用）。"""
    fonts = get_available_japanese_fonts()

    if fonts:
        logger.info(f"利用可能な日本語フォント ({len(fonts)}個):")
        for font in fonts:
            logger.info(f"  - {font}")
    else:
        logger.warning("日本語フォントが見つかりませんでした")
