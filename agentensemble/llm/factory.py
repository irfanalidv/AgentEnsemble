"""
Default LLM provider selection.

Order of precedence:
1. ``AGENTENSEMBLE_LLM_PROVIDER`` env var ("openai" or "mistral"), if set.
2. OpenAI, if ``OPENAI_API_KEY`` is set.
3. Mistral, if ``MISTRAL_API_KEY`` is set.
"""

import os
from typing import Any

from agentensemble.llm.interface import LLMProvider

PROVIDERS = ("openai", "mistral")


def _build(name: str, temperature: float, **kwargs: Any) -> LLMProvider:
    if name == "openai":
        from agentensemble.llm.openai_provider import OpenAILLMProvider

        return OpenAILLMProvider(temperature=temperature, **kwargs)
    if name == "mistral":
        from agentensemble.llm.mistral_provider import MistralLLMProvider

        return MistralLLMProvider(temperature=temperature, **kwargs)
    raise ValueError(f"Unknown LLM provider: {name!r}. Supported: {', '.join(PROVIDERS)}")


def resolve_provider_name() -> str | None:
    """Return the provider that ``get_default_provider`` would use, or None if no key is set."""
    explicit = os.getenv("AGENTENSEMBLE_LLM_PROVIDER", "").strip().lower()
    if explicit:
        return explicit
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("MISTRAL_API_KEY"):
        return "mistral"
    return None


def get_default_provider(
    temperature: float = 0.2, provider: str | None = None, **kwargs: Any
) -> LLMProvider:
    """
    Build the default LLM provider from the environment.

    Args:
        temperature: Sampling temperature passed to the provider.
        provider: Force a provider ("openai" or "mistral") instead of auto-detecting.
        **kwargs: Passed to the provider constructor (e.g. ``model=``).

    Raises:
        ValueError: If no provider can be determined or its API key is missing.
        ImportError: If the provider's LangChain integration is not installed.
    """
    name = (provider or resolve_provider_name() or "").lower()
    if not name:
        raise ValueError(
            "No LLM provider configured. Set OPENAI_API_KEY or MISTRAL_API_KEY "
            "(or AGENTENSEMBLE_LLM_PROVIDER), or pass llm= explicitly."
        )
    return _build(name, temperature, **kwargs)
