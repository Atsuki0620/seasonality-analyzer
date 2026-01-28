"""
LLM Client Builder for Business-flow-maker.

This module provides:
- LLM provider auto-detection (Azure OpenAI / OpenAI)
- Client factory and Protocol definition
- Environment validation utilities
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Protocol

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv() -> None:  # type: ignore
        return None

load_dotenv()

try:
    from openai import AzureOpenAI, OpenAI  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    AzureOpenAI = None  # type: ignore
    OpenAI = None  # type: ignore


class LLMClient(Protocol):
    """Interface for LLM backends."""

    def structured_flow(self, *, prompt: str, schema: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Return a JSON document that satisfies the given schema."""


def cleanup_dummy_proxies() -> None:
    """HTTP(S)_PROXY がダミー値なら環境変数から除去する。"""

    for key in ("HTTP_PROXY", "HTTPS_PROXY"):
        value = os.getenv(key)
        if value and is_dummy_value(value):
            os.environ.pop(key, None)
            print(f"[layer1] {key} はダミー値のため無効化しました。")


def is_dummy_value(value: str) -> bool:
    """ダミー値判定（XXX, your-, 10文字未満など）"""

    if not value:
        return True

    stripped = value.strip()
    if len(stripped) < 10:
        return True

    lower = stripped.lower()
    dummy_tokens = ("xxx", "your-")
    return any(token in lower for token in dummy_tokens)


def validate_openai_env() -> bool:
    """OPENAI_API_KEYが有効か"""

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("[layer1] OpenAI: OPENAI_API_KEY が未設定です。")
        return False
    if is_dummy_value(api_key):
        print("[layer1] OpenAI: OPENAI_API_KEY がダミー値のため利用できません。")
        return False
    return True


def validate_azure_env() -> bool:
    """Azure必須3変数が有効か"""

    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "").strip()
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()

    missing = []
    if not api_key:
        missing.append("AZURE_OPENAI_API_KEY")
    if not api_version:
        missing.append("AZURE_OPENAI_API_VERSION")
    if not endpoint:
        missing.append("AZURE_OPENAI_ENDPOINT")

    if missing:
        print(f"[layer1] AzureOpenAI: {', '.join(missing)} が未設定です。")
        return False

    if any(is_dummy_value(value) for value in (api_key, api_version, endpoint)):
        print("[layer1] AzureOpenAI: 必須環境変数にダミー値が含まれています。")
        return False

    return True


def test_openai_available() -> bool:
    """OpenAI SDKの初期化テスト（軽量、API呼び出しなし）"""

    if OpenAI is None:
        print("[layer1] OpenAI SDK がインストールされていません。`pip install openai` を実行してください。")
        return False

    try:
        OpenAI()
    except Exception as exc:  # pragma: no cover - SDK内部の例外をそのまま通知
        print(f"[layer1] OpenAI クライアント初期化に失敗しました: {exc}")
        return False
    return True


def test_azure_available() -> bool:
    """AzureOpenAI SDKの初期化テスト（軽量、API呼び出しなし）"""

    if AzureOpenAI is None:
        print("[layer1] AzureOpenAI 用の openai SDK がインストールされていません。`pip install openai` を実行してください。")
        return False

    try:
        AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        )
    except Exception as exc:  # pragma: no cover - SDK内部の例外をそのまま通知
        print(f"[layer1] AzureOpenAI クライアント初期化に失敗しました: {exc}")
        return False
    return True


_PROVIDER_CACHE: Optional[str] = None
_PROVIDER_ERRORS: List[str] = []


def detect_provider() -> Optional[str]:
    """
    "azure" | "openai" | None を返す
    Azure優先、標準出力にログ出力
    """

    global _PROVIDER_CACHE, _PROVIDER_ERRORS

    if _PROVIDER_CACHE is not None:
        return _PROVIDER_CACHE

    errors: List[str] = []
    cleanup_dummy_proxies()

    if validate_azure_env():
        if test_azure_available():
            print("[layer1] Azure OpenAI を使用します。")
            _PROVIDER_CACHE = "azure"
            _PROVIDER_ERRORS = []
            return _PROVIDER_CACHE
        errors.append("AzureOpenAI: SDK 初期化に失敗しました。")
    else:
        errors.append("AzureOpenAI: 必須環境変数(AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_ENDPOINT)を正しく設定してください。")

    if validate_openai_env():
        if test_openai_available():
            print("[layer1] OpenAI API を使用します。")
            _PROVIDER_CACHE = "openai"
            _PROVIDER_ERRORS = []
            return _PROVIDER_CACHE
        errors.append("OpenAI: SDK 初期化に失敗しました。")
    else:
        errors.append("OpenAI: OPENAI_API_KEY が未設定かダミー値です。")

    _PROVIDER_CACHE = None
    _PROVIDER_ERRORS = errors

    print("[layer1] 有効な LLM プロバイダを検出できませんでした。")
    for message in errors:
        print(f" - {message}")

    return None


def _extract_json_payload(text: str) -> str:
    """Markdownコードブロックを除去してJSONペイロードを抽出する。"""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped[3:]
        if stripped.startswith("json"):
            stripped = stripped[4:]
        end_idx = stripped.rfind("```")
        if end_idx != -1:
            stripped = stripped[:end_idx]
    return stripped.strip()


class OpenAILLMClient:
    """OpenAI Responses API wrapper that requests JSON-schema constrained output."""

    def __init__(self) -> None:
        if OpenAI is None:
            raise ImportError("openai SDK is not installed. Run `pip install openai`.")
        self._client = OpenAI()

    def structured_flow(self, *, prompt: str, schema: Dict[str, Any], model: str) -> Dict[str, Any]:
        request_kwargs = {
            "model": model,
            "input": prompt,
        }
        try:
            response = self._client.responses.create(
                response_format={"type": "json_schema", "json_schema": {"name": "FlowSchema", "schema": schema}},
                **request_kwargs,
            )
        except TypeError:
            # Older openai SDKs (< 1.43) do not support response_format.
            response = self._client.responses.create(**request_kwargs)
        content = response.output[0].content[0].text if response.output else "{}"
        payload = _extract_json_payload(content)
        return json.loads(payload)


class AzureOpenAILLMClient:
    """AzureOpenAI用クライアント（LLMClient Protocolに準拠）"""

    def __init__(self) -> None:
        if AzureOpenAI is None:
            raise ImportError("openai SDK がインストールされていません。`pip install openai` を実行してください。")

        api_key = os.environ["AZURE_OPENAI_API_KEY"]
        api_version = os.environ["AZURE_OPENAI_API_VERSION"]
        endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]

        self._client = AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint)

    def structured_flow(self, *, prompt: str, schema: Dict[str, Any], model: str) -> Dict[str, Any]:
        request_kwargs = {
            "model": model,
            "input": prompt,
        }
        response = self._client.responses.create(
            response_format={"type": "json_schema", "json_schema": {"name": "FlowSchema", "schema": schema}},
            **request_kwargs,
        )
        content = response.output[0].content[0].text if response.output else "{}"
        payload = _extract_json_payload(content)
        return json.loads(payload)


def create_llm_client() -> LLMClient:
    """detect_provider()の結果に基づきクライアント生成"""

    provider = detect_provider()
    if provider == "azure":
        return AzureOpenAILLMClient()
    if provider == "openai":
        return OpenAILLMClient()

    hint = "\n".join(f"- {message}" for message in _PROVIDER_ERRORS) or "- LLM プロバイダ用の環境変数が不足しています。"
    raise RuntimeError(f"LLM プロバイダを自動検出できませんでした。\n{hint}")
