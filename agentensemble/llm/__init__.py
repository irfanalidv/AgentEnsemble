"""
LLM Provider Interface

Provider-agnostic LLM abstraction for agent reasoning and tool calling.
Ships OpenAI and Mistral providers; any LangChain chat model can be wrapped the same way.

Architecture: Clean separation between provider-specific implementations
and agent logic. Enables multi-provider support without vendor lock-in.
"""

from agentensemble.llm.factory import get_default_provider, resolve_provider_name
from agentensemble.llm.interface import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
    ToolCall,
    ToolSchema,
)

try:
    from agentensemble.llm.mistral_provider import MistralLLMProvider

    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False
    MistralLLMProvider = None

try:
    from agentensemble.llm.openai_provider import OpenAILLMProvider

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAILLMProvider = None

__all__ = [
    "LLMProvider",
    "LLMMessage",
    "ToolCall",
    "LLMResponse",
    "ToolSchema",
    "MistralLLMProvider",
    "MISTRAL_AVAILABLE",
    "OpenAILLMProvider",
    "OPENAI_AVAILABLE",
    "get_default_provider",
    "resolve_provider_name",
]
