"""
OpenAI LLM Provider

Implements LLMProvider using OpenAI chat models via LangChain.
Uses bind_tools for native function calling.
"""

import os
from typing import Any

from agentensemble.llm.langchain_base import LangChainChatProvider

try:
    from langchain_openai import ChatOpenAI
    from openai import DefaultAsyncHttpxClient

    LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    LANGCHAIN_OPENAI_AVAILABLE = False
    ChatOpenAI = None  # type: ignore[assignment,misc]
    DefaultAsyncHttpxClient = None  # type: ignore[assignment,misc]

DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"


def _supports_temperature(model: str) -> bool:
    """Reasoning models (o-series, gpt-5 family) only accept the default temperature."""
    name = model.lower()
    return not (name.startswith(("o1", "o3", "o4", "gpt-5")))


class OpenAILLMProvider(LangChainChatProvider):
    """OpenAI provider using LangChain's ChatOpenAI."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.2,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize OpenAI provider.

        Args:
            model: Model name (e.g., gpt-4.1-mini, gpt-4o-mini, gpt-5-mini). Defaults to the
                ``OPENAI_MODEL`` env var, else ``gpt-4.1-mini``.
            temperature: Lower = more deterministic. Ignored for reasoning models
                (o-series, gpt-5 family), which only support the default.
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env)
            base_url: Optional OpenAI-compatible endpoint (defaults to OPENAI_BASE_URL env)
            **kwargs: Passed to ChatOpenAI
        """
        if not LANGCHAIN_OPENAI_AVAILABLE:
            raise ImportError("langchain-openai required. Install: pip install langchain-openai")

        model = model or os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY required. Set in .env or pass api_key.")

        options: dict[str, Any] = {"model": model, "api_key": key, **kwargs}
        if _supports_temperature(model):
            options["temperature"] = temperature
        base_url = base_url or os.getenv("OPENAI_BASE_URL")
        if base_url:
            options["base_url"] = base_url

        self._options = options
        self.model_name = model
        self._init_model()

    def _build_model(self) -> Any:
        options = dict(self._options)
        # langchain-openai shares one cached async HTTP client per process; it breaks once
        # the event loop it was first used on closes. Give each model its own client.
        options.setdefault("http_async_client", DefaultAsyncHttpxClient())
        return ChatOpenAI(**options)
