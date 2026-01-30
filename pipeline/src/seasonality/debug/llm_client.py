"""診断分析のためのLLMクライアント"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from loguru import logger

from seasonality.debug.bundler import DebugBundle
from seasonality.debug.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from seasonality.config import DebugConfig

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import AzureOpenAI, OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    AzureOpenAI = None
    OpenAI = None
    OPENAI_AVAILABLE = False


@dataclass
class DiagnosticResult:
    """LLM診断分析の結果"""

    summary: str
    issues_found: List[Dict[str, Any]]
    recommendations: List[str]
    confidence: str  # 'high', 'medium', 'low'
    raw_response: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "summary": self.summary,
            "issues_found": self.issues_found,
            "recommendations": self.recommendations,
            "confidence": self.confidence,
        }


class LLMClient(Protocol):
    """LLMクライアントのプロトコル"""

    def diagnose(self, bundle: DebugBundle) -> DiagnosticResult:
        """デバッグバンドルを分析して診断結果を返す"""
        ...


def _is_dummy_value(value: str) -> bool:
    """値がダミー/プレースホルダーかどうかをチェック"""
    if not value:
        return True
    stripped = value.strip()
    if len(stripped) < 10:
        return True
    lower = stripped.lower()
    return any(token in lower for token in ("xxx", "your-", "placeholder"))


def _validate_azure_env() -> bool:
    """Azure OpenAI環境変数を検証"""
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "").strip()
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()

    if not all([api_key, api_version, endpoint]):
        return False

    return not any(_is_dummy_value(v) for v in [api_key, api_version, endpoint])


def _validate_openai_env() -> bool:
    """OpenAI環境変数を検証"""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    return bool(api_key) and not _is_dummy_value(api_key)


def detect_provider() -> Optional[str]:
    """利用可能なLLMプロバイダーを検出

    Returns:
        'azure'、'openai'、またはNone
    """
    if not OPENAI_AVAILABLE:
        logger.warning("openaiパッケージがインストールされていません")
        return None

    if _validate_azure_env():
        logger.info("Azure OpenAIを使用します")
        return "azure"

    if _validate_openai_env():
        logger.info("OpenAIを使用します")
        return "openai"

    logger.warning("有効なLLMプロバイダーが見つかりませんでした")
    return None


class DiagnosticLLMClient:
    """診断分析のためのLLMクライアント"""

    def __init__(self, config: Optional[DebugConfig] = None):
        """診断クライアントを初期化

        Args:
            config: デバッグ設定
        """
        self.config = config or DebugConfig()
        self._client = None
        self._provider = None

        self._initialize_client()

    def _initialize_client(self) -> None:
        """適切なLLMクライアントを初期化"""
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI SDKが利用できません")
            return

        if self.config.provider == "auto":
            self._provider = detect_provider()
        else:
            self._provider = self.config.provider

        if self._provider == "azure":
            self._client = AzureOpenAI(
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            )
        elif self._provider == "openai":
            self._client = OpenAI()
        else:
            logger.warning("利用可能なLLMプロバイダーがありません")

    @property
    def is_available(self) -> bool:
        """クライアントが利用可能かどうかをチェック"""
        return self._client is not None

    def diagnose(self, bundle: DebugBundle) -> DiagnosticResult:
        """デバッグバンドルを分析して診断結果を返す

        Args:
            bundle: 分析するデバッグバンドル

        Returns:
            分析結果を含むDiagnosticResult
        """
        if not self.is_available:
            return DiagnosticResult(
                summary="LLM診断が利用できません",
                issues_found=[],
                recommendations=["詳細な診断にはLLMプロバイダー（Azure OpenAIまたはOpenAI）を設定してください"],
                confidence="low",
            )

        # プロンプトを準備
        bundle_json = json.dumps(bundle.to_dict(), indent=2, ensure_ascii=False)
        user_prompt = USER_PROMPT_TEMPLATE.format(bundle_json=bundle_json)

        try:
            # モデル名を取得
            if self._provider == "azure":
                model = self.config.azure_openai.deployment_name
            else:
                model = self.config.openai.model

            # API呼び出し
            response = self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=2000,
            )

            raw_response = response.choices[0].message.content

            # レスポンスを解析
            return self._parse_response(raw_response)

        except Exception as e:
            logger.error(f"LLM診断が失敗しました: {e}")
            return DiagnosticResult(
                summary=f"診断分析が失敗しました: {str(e)}",
                issues_found=[],
                recommendations=["LLM設定を確認して再試行してください"],
                confidence="low",
                raw_response=str(e),
            )

    def _parse_response(self, response: str) -> DiagnosticResult:
        """LLMレスポンスを構造化された結果に解析"""
        # 最初にJSONとして解析を試みる
        try:
            # レスポンス内のJSONブロックを探す
            import re
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                data = json.loads(json_match.group())
                return DiagnosticResult(
                    summary=data.get("summary", "サマリーが提供されていません"),
                    issues_found=data.get("issues", []),
                    recommendations=data.get("recommendations", []),
                    confidence=data.get("confidence", "medium"),
                    raw_response=response,
                )
        except json.JSONDecodeError:
            pass

        # フォールバック: テキストセクションを抽出
        return DiagnosticResult(
            summary=response[:500] if len(response) > 500 else response,
            issues_found=[],
            recommendations=[],
            confidence="low",
            raw_response=response,
        )


def create_diagnostic_client(config: Optional[DebugConfig] = None) -> DiagnosticLLMClient:
    """診断クライアントを作成するファクトリー関数

    Args:
        config: デバッグ設定

    Returns:
        設定済みのDiagnosticLLMClient
    """
    return DiagnosticLLMClient(config=config)
