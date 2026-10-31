"""
Helpers that let the examples run without an API key.

``RuleBasedLLM`` is a tiny, deterministic ``LLMProvider``: a ``plan`` function picks the tool
call for a question, and an ``answer`` function writes the reply from the tool results. It
implements the same interface as ``OpenAILLMProvider`` / ``MistralLLMProvider``, so the agent
code in each example is identical whether it runs offline or against a real model.

``choose_llm`` returns the real provider when ``OPENAI_API_KEY`` or ``MISTRAL_API_KEY`` is set
(OpenAI first), unless ``--offline`` is on the command line.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

from agentensemble.llm import LLMMessage, LLMProvider, LLMResponse, ToolCall, ToolSchema
from agentensemble.llm.factory import get_default_provider, resolve_provider_name

Plan = Callable[[str, list[ToolSchema]], tuple[str, dict[str, Any]] | None]
Answer = Callable[[str, list[str]], str]


class RuleBasedLLM(LLMProvider):
    """Deterministic stand-in for a chat model, driven by two plain functions."""

    model_name = "rule-based-offline"

    def __init__(self, plan: Plan, answer: Answer) -> None:
        self.plan = plan
        self.answer = answer
        self._calls = 0

    def _respond(self, messages: list[LLMMessage], tools: list[ToolSchema] | None) -> LLMResponse:
        turn_start = max(i for i, m in enumerate(messages) if m.role == "user")
        question = str(messages[turn_start].content)
        results = [str(m.content) for m in messages[turn_start:] if m.role == "tool"]
        self._calls += 1
        if tools and not results:
            step = self.plan(question, tools)
            if step is not None:
                name, args = step
                call = ToolCall(id=f"call_{self._calls}", name=name, arguments=args)
                return LLMResponse(tool_calls=[call], finish_reason="tool_calls")
        return LLMResponse(content=self.answer(question, results))

    def generate(self, messages, tools=None, tool_choice=None, **kwargs: Any) -> LLMResponse:
        return self._respond(messages, tools)

    async def agenerate(self, messages, tools=None, tool_choice=None, **kwargs: Any):
        return self._respond(messages, tools)


def choose_llm(offline: LLMProvider) -> LLMProvider:
    """Use a real provider when a key is configured, otherwise the offline stand-in."""
    if "--offline" in sys.argv or resolve_provider_name() is None:
        print(f"[using offline {type(offline).__name__}; set OPENAI_API_KEY for a real model]\n")
        return offline
    llm = get_default_provider(temperature=0.0)
    print(f"[using {type(llm).__name__} / {getattr(llm, 'model_name', '?')}]\n")
    return llm
