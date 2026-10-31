"""ReActAgent loop behaviour: sessions, tracing, cost, human-in-the-loop, streaming, errors."""

import asyncio

import pytest

from agentensemble import ReActAgent, function_tool
from agentensemble.llm.interface import LLMResponse, ToolCall
from agentensemble.memory import InMemorySession, SQLiteSession
from agentensemble.runner import RunConfig, Runner
from agentensemble.tracing import TraceEventType, TraceHooks
from tests.fakes import ScriptedLLM


@function_tool(description="Look up an order status by id")
def order_status(order_id: str) -> str:
    return {"A1": "shipped", "B2": "processing"}.get(order_id, "unknown")


@function_tool(description="Always fails")
def broken(query: str) -> str:
    raise RuntimeError("backend down")


def tool_then_answer(messages, tools):
    """Call order_status once per turn, then answer from the tool result."""
    turn_start = max(i for i, m in enumerate(messages) if m.role == "user")
    tool_msgs = [m for m in messages[turn_start:] if m.role == "tool"]
    if tools and not tool_msgs:
        return LLMResponse(
            tool_calls=[ToolCall(id="c1", name="order_status", arguments={"order_id": "A1"})],
            usage={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        )
    return LLMResponse(
        content=f"Order A1 is {tool_msgs[-1].content}." if tool_msgs else "Hello again.",
        usage={"input_tokens": 50, "output_tokens": 10, "total_tokens": 60},
    )


class TestSessionPersistence:
    def test_turn_saves_user_query_tool_exchange_and_final_answer(self):
        session = InMemorySession("s")
        agent = ReActAgent(tools=[order_status], llm=ScriptedLLM(responder=tool_then_answer))
        agent.session = session

        result = agent.run("Where is order A1?")

        assert result["result"] == "Order A1 is shipped."
        roles = [m["role"] for m in session.get_messages()]
        assert roles == ["user", "assistant", "tool", "assistant"]
        saved = session.get_messages()
        assert saved[0]["content"] == "Where is order A1?"
        assert saved[-1]["content"] == "Order A1 is shipped."

    def test_tools_remain_available_on_later_turns(self):
        llm = ScriptedLLM(responder=tool_then_answer)
        agent = ReActAgent(tools=[order_status], llm=llm, session=InMemorySession("s"))

        agent.run("Where is order A1?")
        calls_before = len(llm.calls)
        agent.run("And again, where is A1?")

        second_turn_first_call = llm.calls[calls_before]
        assert second_turn_first_call["tools"], "tools must be offered on the second turn"
        # History from turn one is sent along with the new question.
        contents = [m.content for m in second_turn_first_call["messages"]]
        assert "Where is order A1?" in contents
        assert "Order A1 is shipped." in contents

    def test_history_window_never_starts_with_orphan_tool_message(self):
        session = InMemorySession("s")
        session.add_messages(
            [
                {"role": "user", "content": "q1"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "x", "name": "t", "arguments": {}}],
                },
                {"role": "tool", "content": "r", "tool_call_id": "x"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
            ]
        )
        agent = ReActAgent(llm=ScriptedLLM(), session=session)
        msgs = agent._session_to_messages(limit=4)  # window starts at the tool message
        assert [m.role for m in msgs] == ["user", "assistant"]
        assert msgs[0].content == "q2"

    def test_runner_session_is_used_by_agent(self, tmp_path):
        session = SQLiteSession("runner", db_path=str(tmp_path / "s.db"))
        agent = ReActAgent(tools=[order_status], llm=ScriptedLLM(responder=tool_then_answer))

        Runner.run(agent, "Where is order A1?", config=RunConfig(session=session))

        assert agent.session is None  # the agent instance is not mutated
        assert session.get_messages()[0]["role"] == "user"
        assert session.get_messages()[-1]["content"] == "Order A1 is shipped."


class TestTracingAndCost:
    def test_trace_events_usage_and_cost(self):
        hooks = TraceHooks()
        agent = ReActAgent(tools=[order_status], llm=ScriptedLLM(responder=tool_then_answer))

        result = Runner.run(agent, "Where is A1?", config=RunConfig(trace_hooks=hooks))

        types = [e.type for e in hooks.events]
        assert types[0] == TraceEventType.RUN_START
        assert types[-1] == TraceEventType.RUN_END
        assert TraceEventType.TOOL_START in types and TraceEventType.TOOL_END in types
        assert types.count(TraceEventType.LLM_END) == 2
        meta = result["metadata"]
        assert meta["total_usage"] == {
            "input_tokens": 150,
            "output_tokens": 30,
            "total_tokens": 180,
        }
        # gpt-4.1-mini list price: $0.40 / $1.60 per 1M tokens
        assert meta["cost_usd"] == pytest.approx((150 * 0.4 + 30 * 1.6) / 1_000_000)


class TestHumanInTheLoop:
    def test_interrupt_then_resume(self):
        llm = ScriptedLLM(responder=tool_then_answer)
        agent = ReActAgent(tools=[order_status], llm=llm)
        config = RunConfig(interrupt_before_tools=["order_status"])

        paused = Runner.run(agent, "Where is A1?", config=config)
        assert paused["__interrupted__"] is True
        assert paused["pending_tool"] == {"name": "order_status", "arguments": {"order_id": "A1"}}

        final = Runner.run(
            agent,
            "Where is A1?",
            config=config,
            resume=paused["run_state"],
            resume_value="delayed",
        )
        assert final["result"] == "Order A1 is delayed."


class TestToolErrors:
    def test_unknown_tool_and_failing_tool_are_reported_to_the_model(self):
        llm = ScriptedLLM(
            [
                LLMResponse(
                    tool_calls=[
                        ToolCall(id="1", name="missing", arguments={}),
                        ToolCall(id="2", name="broken", arguments={"query": "x"}),
                    ]
                ),
                LLMResponse(content="Sorry, tools failed."),
            ]
        )
        agent = ReActAgent(tools=[broken], llm=llm)
        result = agent.run("hi")
        tool_msgs = [m.content for m in llm.calls[1]["messages"] if m.role == "tool"]
        assert tool_msgs == ["Tool 'missing' not found", "Error: backend down"]
        assert result["result"] == "Sorry, tools failed."
        assert result["metadata"]["tool_calls"] == 2

    def test_run_without_llm_raises(self, monkeypatch):
        for var in ("OPENAI_API_KEY", "MISTRAL_API_KEY", "AGENTENSEMBLE_LLM_PROVIDER"):
            monkeypatch.delenv(var, raising=False)
        agent = ReActAgent()
        with pytest.raises(RuntimeError, match="LLM required"):
            agent.run("hi")


class TestStreaming:
    def test_astream_event_sequence(self):
        agent = ReActAgent(tools=[order_status], llm=ScriptedLLM(responder=tool_then_answer))

        async def collect():
            return [e async for e in agent.astream("Where is A1?")]

        events = asyncio.run(collect())
        assert [e["type"] for e in events] == [
            "thinking",
            "tool_start",
            "tool_end",
            "thinking",
            "done",
        ]
        assert events[2]["result"] == "shipped"
        assert events[-1]["result"] == "Order A1 is shipped."

    def test_astream_without_llm_raises(self, monkeypatch):
        for var in ("OPENAI_API_KEY", "MISTRAL_API_KEY", "AGENTENSEMBLE_LLM_PROVIDER"):
            monkeypatch.delenv(var, raising=False)
        agent = ReActAgent()

        async def consume():
            async for _ in agent.astream("hi"):
                pass

        with pytest.raises(RuntimeError):
            asyncio.run(consume())
