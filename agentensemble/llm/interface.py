"""
LLM Provider Interface

Defines the contract for LLM providers. All providers (Mistral, OpenAI,
Anthropic, local) implement this interface for consistent agent behavior.

Research basis:
- ReAct pattern: Thought -> Action -> Observation (Yao et al., 2022)
- Native tool/function calling preferred over text parsing (LangChain 2024)
- Provider-agnostic design enables multi-model support
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# Message & Response Models
# -----------------------------------------------------------------------------


class LLMMessage(BaseModel):
    """Standardized message format across providers."""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: Union[str, List[Dict[str, Any]]] = ""
    """Content: string for text, or list for multimodal (e.g., tool results)."""
    tool_call_id: Optional[str] = None
    """Required for tool role: links result to specific tool call."""
    tool_calls: Optional[List["ToolCall"]] = None
    """For assistant role: tool invocations when model requests function calls."""


class ToolCall(BaseModel):
    """Structured tool invocation from LLM response."""

    id: str = ""
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """Unified response from any LLM provider."""

    content: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)
    finish_reason: str = "stop"
    """One of: stop, tool_calls, length, content_filter."""
    usage: Optional[Dict[str, int]] = None
    """Token usage: input_tokens, output_tokens."""


class ToolSchema(BaseModel):
    """JSON Schema for function calling (OpenAI/Anthropic compatible)."""

    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    """JSON Schema for parameters."""


# -----------------------------------------------------------------------------
# LLM Provider Protocol
# -----------------------------------------------------------------------------


class LLMProvider(ABC):
    """
    Abstract base for LLM providers.

    Implementations must support:
    - Chat completion with optional tool binding
    - Async execution for non-blocking agent loops
    - Consistent message/response format
    """

    @abstractmethod
    def generate(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[ToolSchema]] = None,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Synchronous generation.

        Args:
            messages: Conversation history
            tools: Available tools (schema format)
            tool_choice: "auto" | "required" | "none" | {"type": "function", "function": {"name": "..."}}
            **kwargs: Provider-specific options (temperature, max_tokens, etc.)

        Returns:
            LLMResponse with content and/or tool_calls
        """
        ...

    @abstractmethod
    async def agenerate(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[ToolSchema]] = None,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Async generation. Same contract as generate."""
        ...

    def get_react_system_prompt(
        self,
        tools: List[ToolSchema],
        max_iterations: int,
    ) -> str:
        """
        Build ReAct system prompt per research best practices.

        References:
        - ReAct: Synergizing Reasoning and Acting (Yao et al., 2022)
        - Tool calling over THOUGHT/ACTION text parsing (LangChain 2024)
        """
        tool_descriptions = "\n".join(
            f"- {t.name}: {t.description}" for t in tools
        )
        return f"""You are a helpful assistant that uses tools to answer questions accurately.

## Your Process (ReAct Pattern)
1. **Reason**: Analyze the user's question and decide the best approach.
2. **Act**: If you need external information, call the appropriate tool with correct parameters.
3. **Observe**: Use the tool result to inform your next step.
4. **Respond**: When you have enough information, provide a clear, accurate answer.

## Available Tools
{tool_descriptions}

## Rules
- Use tools when you need current information, search, or external data.
- Provide your final answer directly when you have sufficient information.
- Maximum {max_iterations} tool calls. Synthesize your answer from available data.
- Be concise and accurate. Cite sources when relevant."""
