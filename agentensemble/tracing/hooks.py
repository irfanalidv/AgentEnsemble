"""
Trace Hooks for Observability

Lightweight event hooks for LLM calls, tool invocations, and agent runs.
No external dependencies; integrates with RunHooks for production use.
Uses provider usage data when available; optional cost estimation via estimate_cost().
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

# Standard pricing per 1M tokens (input, output) - override via estimate_cost(pricing=...)
DEFAULT_PRICING: Dict[str, tuple] = {
    "mistral-large-latest": (2.0, 6.0),
    "mistral-small-latest": (0.35, 0.56),
    "mistral-large": (2.0, 6.0),
    "mistral-small": (0.35, 0.56),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "claude-3-5-sonnet": (3.0, 15.0),
}


def estimate_cost(
    usage: Dict[str, int],
    model: str = "mistral-large-latest",
    pricing: Optional[Dict[str, tuple]] = None,
) -> float:
    """
    Estimate cost in USD from token usage. Uses provider pricing when available.

    Args:
        usage: {"input_tokens": N, "output_tokens": M} (from LLMResponse.usage)
        model: Model name for pricing lookup
        pricing: Optional override {(model, (input_per_1M, output_per_1M))}

    Returns:
        Estimated cost in USD
    """
    if not usage:
        return 0.0
    prices = (pricing or DEFAULT_PRICING).get(model) or DEFAULT_PRICING.get("mistral-large-latest", (2.0, 6.0))
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
    agent: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.perf_counter)
    duration_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
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

    on_event: Optional[Callable[[TraceEvent], None]] = None
    events: List[TraceEvent] = field(default_factory=list)

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
    result: Dict[str, Any],
    hooks: Optional[TraceHooks] = None,
) -> None:
    """Emit run_end trace after agent execution."""
    if hooks:
        data: Dict[str, Any] = {"query": query[:200], "result_preview": str(result.get("result", ""))[:200]}
        if result.get("metadata", {}).get("total_usage"):
            data["usage"] = result["metadata"]["total_usage"]
        hooks.emit(
            TraceEvent(
                type=TraceEventType.RUN_END,
                agent=agent_name,
                data=data,
            )
        )
