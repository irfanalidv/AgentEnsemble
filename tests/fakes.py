"""Deterministic test doubles shared by the test suite (no network, no API keys)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agentensemble.llm.interface import LLMMessage, LLMProvider, LLMResponse, ToolSchema

Responder = Callable[[list[LLMMessage], list[ToolSchema] | None], LLMResponse]


class ScriptedLLM(LLMProvider):
    """Returns queued responses in order (or calls a responder) and records every request."""

    model_name = "gpt-4.1-mini"

    def __init__(
        self,
        responses: list[LLMResponse] | None = None,
        responder: Responder | None = None,
    ) -> None:
        self.responses = list(responses or [])
        self.responder = responder
        self.calls: list[dict[str, Any]] = []

    def _next(self, messages: list[LLMMessage], tools: list[ToolSchema] | None) -> LLMResponse:
        self.calls.append({"messages": list(messages), "tools": tools})
        if self.responder is not None:
            return self.responder(messages, tools)
        if not self.responses:
            raise AssertionError("ScriptedLLM ran out of responses")
        return self.responses.pop(0)

    def generate(self, messages, tools=None, tool_choice=None, **kwargs):
        return self._next(messages, tools)

    async def agenerate(self, messages, tools=None, tool_choice=None, **kwargs):
        return self._next(messages, tools)
