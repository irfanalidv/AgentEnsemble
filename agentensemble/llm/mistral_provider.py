"""
Mistral LLM Provider

Implements LLMProvider using Mistral AI via LangChain.
Uses bind_tools for native function calling (preferred over text parsing).
"""

import os
from typing import Any

from agentensemble.llm.langchain_base import (  # noqa: F401  (re-exported for compatibility)
    LangChainChatProvider,
    _extract_usage,
    _messages_to_langchain,
    _parse_ai_message,
    _tool_schema_to_langchain,
)

try:
    from langchain_mistralai import ChatMistralAI

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    ChatMistralAI = None


class MistralLLMProvider(LangChainChatProvider):
    """
    Mistral AI provider using LangChain's ChatMistralAI.

    Uses bind_tools for native function calling—cleaner than
    THOUGHT/ACTION/PAUSE text parsing (LangChain 2024 best practice).
    """

    # Mistral spells "must call a tool" as "any".
    REQUIRED_TOOL_CHOICE = "any"

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.2,
        api_key: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize Mistral provider.

        Args:
            model: Model name (e.g., mistral-large-latest, mistral-small-latest). Defaults to
                the ``MISTRAL_MODEL`` env var, else ``mistral-large-latest``.
            temperature: Lower = more deterministic (0.2 recommended for tool use)
            api_key: Mistral API key (defaults to MISTRAL_API_KEY env)
            **kwargs: Passed to ChatMistralAI
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-mistralai and langchain-core required. "
                "Install: pip install langchain-mistralai langchain-core"
            )

        model = model or os.getenv("MISTRAL_MODEL") or "mistral-large-latest"
        key = api_key or os.getenv("MISTRAL_API_KEY")
        if not key:
            raise ValueError("MISTRAL_API_KEY required. Set in .env or pass api_key.")

        self._options: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "mistral_api_key": key,
            **kwargs,
        }
        self.model_name = model
        self._init_model()

    def _build_model(self) -> Any:
        return ChatMistralAI(**self._options)
