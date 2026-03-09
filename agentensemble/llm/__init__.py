"""
LLM Provider Interface

Provider-agnostic LLM abstraction for agent reasoning and tool calling.
Supports Mistral, OpenAI, Anthropic, and local models via a unified interface.

Architecture: Clean separation between provider-specific implementations
and agent logic. Enables multi-provider support without vendor lock-in.
"""

from agentensemble.llm.interface import (
    LLMProvider,
    LLMMessage,
    ToolCall,
    LLMResponse,
    ToolSchema,
)

try:
    from agentensemble.llm.mistral_provider import MistralLLMProvider
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False
    MistralLLMProvider = None

__all__ = [
    "LLMProvider",
    "LLMMessage",
    "ToolCall",
    "LLMResponse",
    "ToolSchema",
    "MistralLLMProvider",
    "MISTRAL_AVAILABLE",
]
