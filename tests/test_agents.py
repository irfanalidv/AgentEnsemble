"""Tests for agent implementations."""

import pytest

from agentensemble.agents import ReActAgent, StateGraphAgent, RAGAgent, HybridAgent
from agentensemble.agents.base import BaseAgent, AgentState
from agentensemble.tools import function_tool


class TestAgentState:
    """Tests for AgentState model."""

    def test_agent_state_defaults(self):
        state = AgentState(query="hello")
        assert state.query == "hello"
        assert state.context == {}
        assert state.iteration_count == 0
        assert state.tool_calls == []
        assert state.result is None

    def test_agent_state_with_context(self):
        state = AgentState(query="q", context={"key": "value"})
        assert state.context["key"] == "value"


class TestBaseAgent:
    """Tests for BaseAgent methods."""

    def test_validate_state(self):
        agent = ReActAgent(name="t", tools=[], max_iterations=3)
        state = AgentState(query="q", iteration_count=0)
        assert agent._validate_state(state) is True
        state.iteration_count = 3
        assert agent._validate_state(state) is False

    def test_update_state(self):
        agent = ReActAgent(name="t", tools=[], max_iterations=3)
        state = AgentState(query="q")
        updated = agent._update_state(state, result="done", tool_calls=[{"name": "x"}])
        assert updated.result == "done"
        assert len(updated.tool_calls) == 1
        assert updated.iteration_count == 1


class TestStateGraphAgent:
    """Tests for StateGraphAgent."""

    def test_stategraph_simple_run(self, stategraph_agent):
        # Override route to go start -> end
        def route(state, current):
            return "end" if current == "start" else "end"

        stategraph_agent._route = route
        result = stategraph_agent.run("hello")
        assert "result" in result
        assert "metadata" in result
        assert "Processed:" in result["result"] or "No result" in result["result"]


class TestRAGAgent:
    """Tests for RAGAgent."""

    def test_rag_agent_no_tools(self):
        agent = RAGAgent(name="rag", tools=[], fallback_strategies=2)
        result = agent.run("test query")
        assert "No result" in result["result"] or "RAG tool not available" in result["result"]
        assert "metadata" in result


class TestHybridAgent:
    """Tests for HybridAgent."""

    def test_hybrid_agent_no_tools(self):
        agent = HybridAgent(name="hybrid", tools=[], max_iterations=2, llm=None)
        result = agent.run("test")
        assert "result" in result
        assert "Unable to generate answer" in result["result"] or "No result" in result["result"]

    def test_hybrid_agent_llm_routing_mock(self):
        """Test LLM routing path with mock LLM (no API key needed)."""
        from agentensemble.llm.interface import LLMResponse, ToolCall

        class MockLLM:
            async def agenerate(self, messages, tools=None, tool_choice=None, **kwargs):
                return LLMResponse(
                    content=None,
                    tool_calls=[ToolCall(id="1", name="choose_next_action", arguments={"action": "ANSWER"})],
                )

        agent = HybridAgent(name="hybrid", tools=[], max_iterations=2, llm=MockLLM())
        result = agent.run("test")
        assert "result" in result
        assert "actions_taken" in result["metadata"]
        assert "ANSWER" in result["metadata"]["actions_taken"]


class TestFunctionTool:
    """Tests for @function_tool decorator."""

    def test_function_tool_schema(self):
        @function_tool(description="Test tool")
        def my_tool(x: str) -> str:
            return x

        schema = my_tool.get_schema()
        assert schema.name == "my_tool"
        assert schema.description == "Test tool"
        assert "properties" in schema.parameters

    def test_function_tool_run(self):
        @function_tool()
        def add(a: int, b: int) -> int:
            return a + b

        assert add.run(a=1, b=2) == 3
