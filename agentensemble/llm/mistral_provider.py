"""
Mistral LLM Provider

Implements LLMProvider using Mistral AI via LangChain.
Uses bind_tools for native function calling (preferred over text parsing).
"""

import json
from typing import Any, Dict, List, Optional

from agentensemble.llm.interface import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
    ToolCall,
    ToolSchema,
)

try:
    from langchain_mistralai import ChatMistralAI
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        SystemMessage,
        ToolCall as LCToolCall,
        ToolMessage,
    )
    from langchain_core.runnables import Runnable

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    ChatMistralAI = None
    LCToolCall = None
    Runnable = None


def _tool_schema_to_langchain(schema: ToolSchema) -> Dict[str, Any]:
    """Convert ToolSchema to OpenAI-style format for Mistral."""
    return {
        "type": "function",
        "function": {
            "name": schema.name,
            "description": schema.description,
            "parameters": schema.parameters if schema.parameters else {"type": "object", "properties": {}},
        },
    }


def _messages_to_langchain(messages: List[LLMMessage]) -> List["BaseMessage"]:
    """Convert LLMMessage list to LangChain format."""
    result: List[BaseMessage] = []
    for msg in messages:
        content_str = msg.content if isinstance(msg.content, str) else str(msg.content)
        if msg.role == "system":
            result.append(SystemMessage(content=content_str))
        elif msg.role == "user":
            result.append(HumanMessage(content=content_str))
        elif msg.role == "assistant":
            lc_tool_calls = []
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    lc_tool_calls.append(
                        LCToolCall(
                            id=tc.id or f"call_{tc.name}",
                            type="function",
                            name=tc.name,
                            args=tc.arguments,
                        )
                    )
            result.append(
                AIMessage(content=content_str, tool_calls=lc_tool_calls if lc_tool_calls else [])
            )
        elif msg.role == "tool":
            result.append(
                ToolMessage(
                    content=content_str,
                    tool_call_id=msg.tool_call_id or "",
                )
            )
    return result


def _parse_ai_message(msg: AIMessage) -> LLMResponse:
    """Extract LLMResponse from LangChain AIMessage."""
    content = msg.content if isinstance(msg.content, str) else ""
    tool_calls: List[ToolCall] = []

    if hasattr(msg, "tool_calls") and msg.tool_calls:
        for tc in msg.tool_calls:
            if isinstance(tc, dict):
                args = tc.get("args", tc.get("arguments", {}))
                tc_id = tc.get("id", "")
                tc_name = tc.get("name", "")
            else:
                args = getattr(tc, "args", getattr(tc, "arguments", {}))
                tc_id = getattr(tc, "id", "") or ""
                tc_name = getattr(tc, "name", "") or ""
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append(
                ToolCall(
                    id=tc_id,
                    name=tc_name,
                    arguments=args if isinstance(args, dict) else {},
                )
            )

    return LLMResponse(
        content=content or None,
        tool_calls=tool_calls,
        finish_reason="tool_calls" if tool_calls else "stop",
    )


class MistralLLMProvider(LLMProvider):
    """
    Mistral AI provider using LangChain's ChatMistralAI.

    Uses bind_tools for native function calling—cleaner than
    THOUGHT/ACTION/PAUSE text parsing (LangChain 2024 best practice).
    """

    def __init__(
        self,
        model: str = "mistral-large-latest",
        temperature: float = 0.2,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize Mistral provider.

        Args:
            model: Model name (e.g., mistral-large-latest, mistral-small-latest)
            temperature: Lower = more deterministic (0.2 recommended for tool use)
            api_key: Mistral API key (defaults to MISTRAL_API_KEY env)
            **kwargs: Passed to ChatMistralAI
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-mistralai and langchain-core required. "
                "Install: pip install langchain-mistralai langchain-core"
            )

        import os
        key = api_key or os.getenv("MISTRAL_API_KEY")
        if not key:
            raise ValueError("MISTRAL_API_KEY required. Set in .env or pass api_key.")

        self._model = ChatMistralAI(
            model=model,
            temperature=temperature,
            mistral_api_key=key,
            **kwargs,
        )
        self.model_name = model

    def _bind_tools(
        self,
        tools: Optional[List[ToolSchema]] = None,
        tool_choice: Optional[str] = None,
    ) -> "Runnable":
        """Bind tools to model for function calling."""
        if not tools:
            return self._model

        tool_defs = [_tool_schema_to_langchain(t) for t in tools]
        bound = self._model.bind_tools(tool_defs)

        if tool_choice == "required" and tools:
            # Mistral: use "any" to force tool use (required -> any per Mistral API)
            bound = bound.bind(tool_choice="any")
        elif tool_choice and tool_choice != "auto":
            bound = bound.bind(tool_choice=tool_choice)

        return bound

    def generate(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[ToolSchema]] = None,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Synchronous generation with optional tool calling."""
        lc_messages = _messages_to_langchain(messages)
        bound = self._bind_tools(tools, tool_choice)
        response = bound.invoke(lc_messages, **kwargs)
        return _parse_ai_message(response)

    async def agenerate(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[ToolSchema]] = None,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Async generation with optional tool calling."""
        lc_messages = _messages_to_langchain(messages)
        bound = self._bind_tools(tools, tool_choice)
        response = await bound.ainvoke(lc_messages, **kwargs)
        return _parse_ai_message(response)
