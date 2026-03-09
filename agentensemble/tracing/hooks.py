"""
Trace Hooks for Observability

Lightweight event hooks for LLM calls, tool invocations, and agent runs.
No external dependencies; integrates with RunHooks for production use.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


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
        hooks.emit(
            TraceEvent(
                type=TraceEventType.RUN_END,
                agent=agent_name,
                data={"query": query[:200], "result_preview": str(result.get("result", ""))[:200]},
            )
        )
