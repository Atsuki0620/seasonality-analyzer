"""LLM client for diagnostic analysis."""

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
    """Result of LLM diagnostic analysis."""

    summary: str
    issues_found: List[Dict[str, Any]]
    recommendations: List[str]
    confidence: str  # 'high', 'medium', 'low'
    raw_response: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "summary": self.summary,
            "issues_found": self.issues_found,
            "recommendations": self.recommendations,
            "confidence": self.confidence,
        }


class LLMClient(Protocol):
    """Protocol for LLM clients."""

    def diagnose(self, bundle: DebugBundle) -> DiagnosticResult:
        """Analyze debug bundle and return diagnostic result."""
        ...


def _is_dummy_value(value: str) -> bool:
    """Check if value is a dummy/placeholder."""
    if not value:
        return True
    stripped = value.strip()
    if len(stripped) < 10:
        return True
    lower = stripped.lower()
    return any(token in lower for token in ("xxx", "your-", "placeholder"))


def _validate_azure_env() -> bool:
    """Validate Azure OpenAI environment variables."""
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "").strip()
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()

    if not all([api_key, api_version, endpoint]):
        return False

    return not any(_is_dummy_value(v) for v in [api_key, api_version, endpoint])


def _validate_openai_env() -> bool:
    """Validate OpenAI environment variables."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    return bool(api_key) and not _is_dummy_value(api_key)


def detect_provider() -> Optional[str]:
    """Detect available LLM provider.

    Returns:
        'azure', 'openai', or None.
    """
    if not OPENAI_AVAILABLE:
        logger.warning("openai package not installed")
        return None

    if _validate_azure_env():
        logger.info("Using Azure OpenAI")
        return "azure"

    if _validate_openai_env():
        logger.info("Using OpenAI")
        return "openai"

    logger.warning("No valid LLM provider found")
    return None


class DiagnosticLLMClient:
    """LLM client for diagnostic analysis."""

    def __init__(self, config: Optional[DebugConfig] = None):
        """Initialize diagnostic client.

        Args:
            config: Debug configuration.
        """
        self.config = config or DebugConfig()
        self._client = None
        self._provider = None

        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the appropriate LLM client."""
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI SDK not available")
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
            logger.warning("No LLM provider available")

    @property
    def is_available(self) -> bool:
        """Check if client is available."""
        return self._client is not None

    def diagnose(self, bundle: DebugBundle) -> DiagnosticResult:
        """Analyze debug bundle and return diagnostic result.

        Args:
            bundle: Debug bundle to analyze.

        Returns:
            DiagnosticResult with analysis.
        """
        if not self.is_available:
            return DiagnosticResult(
                summary="LLM diagnostics not available",
                issues_found=[],
                recommendations=["Configure LLM provider (Azure OpenAI or OpenAI) for detailed diagnostics"],
                confidence="low",
            )

        # Prepare prompt
        bundle_json = json.dumps(bundle.to_dict(), indent=2, ensure_ascii=False)
        user_prompt = USER_PROMPT_TEMPLATE.format(bundle_json=bundle_json)

        try:
            # Get model name
            if self._provider == "azure":
                model = self.config.azure_openai.deployment_name
            else:
                model = self.config.openai.model

            # Make API call
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

            # Parse response
            return self._parse_response(raw_response)

        except Exception as e:
            logger.error(f"LLM diagnostic failed: {e}")
            return DiagnosticResult(
                summary=f"Diagnostic analysis failed: {str(e)}",
                issues_found=[],
                recommendations=["Check LLM configuration and try again"],
                confidence="low",
                raw_response=str(e),
            )

    def _parse_response(self, response: str) -> DiagnosticResult:
        """Parse LLM response into structured result."""
        # Try to parse as JSON first
        try:
            # Look for JSON block in response
            import re
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                data = json.loads(json_match.group())
                return DiagnosticResult(
                    summary=data.get("summary", "No summary provided"),
                    issues_found=data.get("issues", []),
                    recommendations=data.get("recommendations", []),
                    confidence=data.get("confidence", "medium"),
                    raw_response=response,
                )
        except json.JSONDecodeError:
            pass

        # Fallback: extract text sections
        return DiagnosticResult(
            summary=response[:500] if len(response) > 500 else response,
            issues_found=[],
            recommendations=[],
            confidence="low",
            raw_response=response,
        )


def create_diagnostic_client(config: Optional[DebugConfig] = None) -> DiagnosticLLMClient:
    """Factory function to create diagnostic client.

    Args:
        config: Debug configuration.

    Returns:
        Configured DiagnosticLLMClient.
    """
    return DiagnosticLLMClient(config=config)
