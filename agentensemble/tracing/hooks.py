"""
Trace Hooks for Observability

Lightweight event hooks for LLM calls, tool invocations, and agent runs.
No external dependencies; integrates with RunHooks for production use.
Uses provider usage data when available; optional cost estimation via estimate_cost().
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Standard list prices in USD per 1M tokens (input, output), checked against the providers'
# official pricing pages on PRICING_AS_OF:
#   OpenAI     https://developers.openai.com/api/docs/pricing
#   Anthropic  https://platform.claude.com/docs/en/about-claude/pricing
#   Mistral    https://docs.mistral.ai/inference/pricing
# Long-context tiers are listed at their short-context price. Prices change; pass
# estimate_cost(pricing=...) for exact numbers. Names match exactly, then by longest prefix.
PRICING_AS_OF = "2026-10-31"

DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-6-astra": (10.0, 50.0),
    "gpt-6.1-sol": (2.0, 10.0),
    "gpt-6.1-terra": (1.0, 5.0),
    "gpt-6-sol": (2.0, 10.0),
    "gpt-6-luna": (0.10, 0.50),
    "gpt-5.6-sol": (4.0, 20.0),
    "gpt-5.6-terra": (2.0, 12.0),
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-5.5": (5.0, 30.0),
    "gpt-5.4": (2.50, 15.0),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5.2": (1.75, 14.0),
    "gpt-5.1": (1.25, 10.0),
    "gpt-5": (1.25, 10.0),
    "gpt-5-mini": (0.25, 2.0),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "o3": (2.0, 8.0),
    "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40),
    # Anthropic
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-5": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4": (3.0, 15.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-3-5-sonnet": (3.0, 15.0),  # retired; last list price
    # Mistral: family prefixes also cover -latest aliases and dated ids
    "mistral-large": (0.70, 2.10),
    "mistral-small": (0.15, 0.60),
    "ministral-14b": (0.20, 0.20),
    "ministral-8b": (0.15, 0.15),
    "ministral-3b": (0.10, 0.10),
    "codestral": (0.30, 0.90),
}


def _lookup_price(model: str, table: dict[str, tuple[float, float]]) -> tuple[float, float] | None:
    """Exact match first, then the longest known prefix (``gpt-4.1-mini-2025-04-14``)."""
    if model in table:
        return table[model]
    for name in sorted(table, key=len, reverse=True):
        if model.startswith(name):
            return table[name]
    return None


def estimate_cost(
    usage: dict[str, int],
    model: str = "mistral-large-latest",
    pricing: dict[str, tuple[float, float]] | None = None,
) -> float:
    """
    Estimate cost in USD from token usage. Uses provider pricing when available.

    Args:
        usage: {"input_tokens": N, "output_tokens": M} (from LLMResponse.usage)
        model: Model name for pricing lookup
        pricing: Optional override {(model, (input_per_1M, output_per_1M))}

    Returns:
        Estimated cost in USD. Unknown models are priced as ``mistral-large``.
    """
    if not usage:
        return 0.0
    prices = (
        _lookup_price(model, pricing or DEFAULT_PRICING)
        or _lookup_price(model, DEFAULT_PRICING)
        or DEFAULT_PRICING["mistral-large"]
    )
    inp = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
    out = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
    return (inp * prices[0] + out * prices[1]) / 1_000_000


class TraceEventType(str, Enum):
    """Event types for tracing."""

    RUN_START = "run_start"
    RUN_END = "run_end"
    LLM_START = "llm_start"
    LLM_END = "llm_end"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    ERROR = "error"


@dataclass
class TraceEvent:
    """Single trace event."""

    type: TraceEventType
    agent: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.perf_counter)
    duration_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "agent": self.agent,
            "data": self.data,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
        }


@dataclass
class TraceHooks:
    """
    Observability hooks for agent execution.

    Attach to RunConfig.hooks or use as standalone.
    """

    on_event: Callable[[TraceEvent], None] | None = None
    events: list[TraceEvent] = field(default_factory=list)

    def emit(self, event: TraceEvent) -> None:
        """Emit event to callback and store."""
        self.events.append(event)
        if self.on_event:
            self.on_event(event)

    def clear(self) -> None:
        """Clear stored events."""
        self.events.clear()


def trace_run(
    agent_name: str,
    query: str,
    result: dict[str, Any],
    hooks: TraceHooks | None = None,
) -> None:
    """Emit run_end trace after agent execution."""
    if hooks:
        data: dict[str, Any] = {
            "query": query[:200],
            "result_preview": str(result.get("result", ""))[:200],
        }
        if result.get("metadata", {}).get("total_usage"):
            data["usage"] = result["metadata"]["total_usage"]
        hooks.emit(
            TraceEvent(
                type=TraceEventType.RUN_END,
                agent=agent_name,
                data=data,
            )
        )
