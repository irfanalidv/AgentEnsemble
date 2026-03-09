"""Tests for orchestration patterns."""

import pytest

from agentensemble.agents import ReActAgent, StateGraphAgent
from agentensemble.agents.base import AgentState
from agentensemble.orchestration import Ensemble, SwarmOrchestrator, PipelineOrchestrator, DebateOrchestrator
from agentensemble.tools import function_tool


@pytest.fixture
def simple_agent():
    """Agent that returns fixed result."""

    @function_tool(description="Echo")
    def echo(x: str) -> str:
        return x

    class EchoAgent(ReActAgent):
        pass

    return ReActAgent(name="echo", tools=[], max_iterations=1)


@pytest.fixture
def mock_agents():
    """Agents that don't need LLM - use StateGraphAgent for simplicity."""

    def node(state):
        return {"result": f"Processed: {state.query}"}

    return {
        "a": StateGraphAgent(
            name="a",
            nodes={"start": node, "end": node},
            max_iterations=2,
        ),
        "b": StateGraphAgent(
            name="b",
            nodes={"start": node, "end": node},
            max_iterations=2,
        ),
    }


class TestEnsemble:
    """Tests for Ensemble."""

    def test_supervisor_perform(self, mock_agents):
        ensemble = Ensemble(agents=mock_agents, conductor="supervisor")
        result = ensemble.perform(task="hello", data={"query": "hello"})
        assert result["conductor"] == "supervisor"
        assert "results" in result
        assert "a" in result["results"]
        assert "b" in result["results"]

    def test_pipeline_perform(self, mock_agents):
        ensemble = Ensemble(agents=mock_agents, conductor="pipeline")
        result = ensemble.perform(task="hello", data={"query": "hello"})
        assert result["conductor"] == "pipeline"
        assert "final_result" in result


class TestSwarmOrchestrator:
    """Tests for SwarmOrchestrator."""

    @pytest.mark.asyncio
    async def test_swarm_aperform(self, mock_agents):
        swarm = SwarmOrchestrator(agents=mock_agents)
        result = await swarm.aperform(task="hello", data={"query": "hello"})
        assert result["conductor"] == "swarm"
        assert len(result["results"]) == 2


class TestDebateOrchestrator:
    """Tests for DebateOrchestrator."""

    @pytest.fixture
    def mock_solvers(self):
        def node(state):
            return {"result": "42"}

        return StateGraphAgent(
            name="solver",
            nodes={"start": node},
            max_iterations=1,
        )

    @pytest.mark.asyncio
    async def test_debate_structure(self, mock_solvers):
        debate = DebateOrchestrator(
            solvers=[mock_solvers],
            aggregator=mock_solvers,
            rounds=1,
        )
        result = await debate.adebate("What is 6*7?")
        assert "result" in result
        assert "proposals" in result
        assert "metadata" in result
