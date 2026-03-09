"""Pytest fixtures for AgentEnsemble tests."""

import pytest

from agentensemble.agents import ReActAgent, StateGraphAgent, RAGAgent, HybridAgent
from agentensemble.agents.base import BaseAgent, AgentState
from agentensemble.tools import function_tool, SearchTool
from agentensemble.memory import InMemorySession
from agentensemble.orchestration import Ensemble, SwarmOrchestrator, PipelineOrchestrator


@pytest.fixture
def agent_state():
    """Sample AgentState for testing."""
    return AgentState(query="test query", context={})


@pytest.fixture
def mock_tool():
    """Mock tool that returns fixed result."""

    @function_tool(description="A test tool")
    def mock_tool_func(query: str) -> str:
        return f"Result for: {query}"

    return mock_tool_func


@pytest.fixture
def react_agent_no_llm(mock_tool):
    """ReActAgent without LLM - for unit tests that mock LLM."""
    return ReActAgent(name="test_react", tools=[mock_tool], max_iterations=2)


@pytest.fixture
def stategraph_agent():
    """StateGraphAgent with simple nodes."""
    def start_node(state):
        return {"context": {**state.context, "started": True}}

    def end_node(state):
        return {"result": f"Processed: {state.query}"}

    return StateGraphAgent(
        name="test_graph",
        nodes={"start": start_node, "end": end_node},
        max_iterations=5,
    )


@pytest.fixture
def in_memory_session():
    """InMemorySession for testing."""
    return InMemorySession(session_id="test-session")
