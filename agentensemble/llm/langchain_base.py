"""
Shared LangChain plumbing for chat-model providers.

Converts between AgentEnsemble's provider-agnostic message/tool models and
LangChain's chat-model API. Concrete providers (Mistral, OpenAI) only need to
build the underlying LangChain chat model.
"""

import asyncio
import json
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages import (
    ToolCall as LCToolCall,
)
from langchain_core.runnables import Runnable

from agentensemble.llm.interface import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
    ToolCall,
    ToolSchema,
)


def _tool_schema_to_langchain(schema: ToolSchema) -> dict[str, Any]:
    """Convert ToolSchema to OpenAI-style function format (also accepted by Mistral)."""
    return {
        "type": "function",
        "function": {
            "name": schema.name,
            "description": schema.description,
            "parameters": schema.parameters
            if schema.parameters
            else {"type": "object", "properties": {}},
        },
    }


def _messages_to_langchain(messages: list[LLMMessage]) -> list[BaseMessage]:
    """Convert LLMMessage list to LangChain format."""
    result: list[BaseMessage] = []
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
                            type="tool_call",
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


def _extract_usage(msg: AIMessage) -> dict[str, int] | None:
    """Extract token usage from LangChain AIMessage (usage_metadata or response_metadata)."""
    um = getattr(msg, "usage_metadata", None)
    if um and isinstance(um, dict):
        return {
            "input_tokens": um.get("input_tokens", 0),
            "output_tokens": um.get("output_tokens", 0),
            "total_tokens": um.get("total_tokens", 0),
        }
    rm = getattr(msg, "response_metadata", None) or {}
    usage = rm.get("usage") or rm.get("usage_metadata") or rm.get("token_usage") or {}
    if usage:
        return {
            "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens", 0)),
            "output_tokens": usage.get("output_tokens", usage.get("completion_tokens", 0)),
            "total_tokens": usage.get("total_tokens", 0),
        }
    return None


def _parse_ai_message(msg: AIMessage) -> LLMResponse:
    """Extract LLMResponse from LangChain AIMessage."""
    content = msg.content if isinstance(msg.content, str) else ""
    tool_calls: list[ToolCall] = []
    usage = _extract_usage(msg)

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
                    id=tc_id or "",
                    name=tc_name,
                    arguments=args if isinstance(args, dict) else {},
                )
            )

    return LLMResponse(
        content=content or None,
        tool_calls=tool_calls,
        finish_reason="tool_calls" if tool_calls else "stop",
        usage=usage,
    )


class LangChainChatProvider(LLMProvider):
    """
    Base provider wrapping any LangChain chat model that supports ``bind_tools``.

    Subclasses implement ``_build_model()``, call ``_init_model()`` at the end of
    ``__init__``, set ``self.model_name``, and may override ``REQUIRED_TOOL_CHOICE``
    with the provider's spelling of "must call a tool".

    Async HTTP clients are bound to the event loop they were first used on. The sync
    agent APIs call ``asyncio.run()`` (a fresh loop per call), so the model is rebuilt
    whenever ``agenerate`` runs on a different loop than last time.
    """

    #: Value sent to the API when callers ask for ``tool_choice="required"``.
    REQUIRED_TOOL_CHOICE: str = "required"

    _model: Any
    _model_loop: asyncio.AbstractEventLoop | None = None
    model_name: str

    def _build_model(self) -> Any:
        """Create the underlying LangChain chat model."""
        raise NotImplementedError

    def _init_model(self) -> None:
        self._model = self._build_model()
        self._model_loop = None

    def _ensure_model_for_running_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._model_loop is loop:
            return
        if self._model_loop is not None:
            self._model = self._build_model()
        self._model_loop = loop

    def _bind_tools(
        self,
        tools: list[ToolSchema] | None = None,
        tool_choice: str | None = None,
    ) -> Runnable:
        """Bind tools to model for function calling."""
        if not tools:
            return self._model

        tool_defs = [_tool_schema_to_langchain(t) for t in tools]
        if tool_choice == "required":
            return self._model.bind_tools(tool_defs, tool_choice=self.REQUIRED_TOOL_CHOICE)
        if tool_choice and tool_choice != "auto":
            return self._model.bind_tools(tool_defs, tool_choice=tool_choice)
        return self._model.bind_tools(tool_defs)

    def generate(
        self,
        messages: list[LLMMessage],
        tools: list[ToolSchema] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Synchronous generation with optional tool calling."""
        lc_messages = _messages_to_langchain(messages)
        bound = self._bind_tools(tools, tool_choice)
        response = bound.invoke(lc_messages, **kwargs)
        return _parse_ai_message(response)

    async def agenerate(
        self,
        messages: list[LLMMessage],
        tools: list[ToolSchema] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Async generation with optional tool calling."""
        self._ensure_model_for_running_loop()
        lc_messages = _messages_to_langchain(messages)
        bound = self._bind_tools(tools, tool_choice)
        response = await bound.ainvoke(lc_messages, **kwargs)
        return _parse_ai_message(response)
