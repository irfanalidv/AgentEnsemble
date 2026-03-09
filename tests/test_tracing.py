"""Tests for tracing and observability."""

import pytest

from agentensemble.tracing import TraceHooks, TraceEvent, TraceEventType, estimate_cost


class TestEstimateCost:
    """Tests for estimate_cost."""

    def test_empty_usage_returns_zero(self):
        assert estimate_cost({}) == 0.0

    def test_usage_calculation(self):
        usage = {"input_tokens": 1000, "output_tokens": 500}
        cost = estimate_cost(usage, model="mistral-large-latest")
        assert cost > 0
        # 1000 * 2/1e6 + 500 * 6/1e6 = 0.002 + 0.003 = 0.005
        assert 0.004 < cost < 0.006

    def test_prompt_completion_aliases(self):
        usage = {"prompt_tokens": 100, "completion_tokens": 50}
        cost = estimate_cost(usage)
        assert cost > 0


class TestTraceHooks:
    """Tests for TraceHooks."""

    def test_emit_stores_events(self):
        hooks = TraceHooks()
        hooks.emit(TraceEvent(type=TraceEventType.RUN_START, agent="test"))
        hooks.emit(TraceEvent(type=TraceEventType.RUN_END, agent="test"))
        assert len(hooks.events) == 2
        assert hooks.events[0].type == TraceEventType.RUN_START
        assert hooks.events[1].type == TraceEventType.RUN_END

    def test_emit_calls_callback(self):
        events = []
        hooks = TraceHooks(on_event=lambda e: events.append(e))
        hooks.emit(TraceEvent(type=TraceEventType.TOOL_START, agent="a", data={"tool": "search"}))
        assert len(events) == 1
        assert events[0].data["tool"] == "search"

    def test_clear(self):
        hooks = TraceHooks()
        hooks.emit(TraceEvent(type=TraceEventType.RUN_START, agent="x"))
        hooks.clear()
        assert len(hooks.events) == 0
