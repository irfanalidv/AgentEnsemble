"""Tests for Runner, RunConfig, RunHooks."""

import pytest

from agentensemble.runner import Runner, RunConfig, RunHooks
from agentensemble.agents import StateGraphAgent


class MockAgent:
    """Mock agent for testing Runner."""

    name = "mock"

    def run(self, query: str, **kwargs):
        return {"result": f"mock: {query}", "metadata": {}}

    async def arun(self, query: str, **kwargs):
        return {"result": f"mock: {query}", "metadata": {}}


class TestRunner:
    """Tests for Runner."""

    def test_run_sync(self):
        agent = MockAgent()
        result = Runner.run(agent, "hello")
        assert result["result"] == "mock: hello"

    @pytest.mark.asyncio
    async def test_arun_async(self):
        agent = MockAgent()
        result = await Runner.arun(agent, "hello")
        assert result["result"] == "mock: hello"

    def test_run_with_config_context(self):
        agent = MockAgent()
        config = RunConfig(context={"extra": "value"})
        result = Runner.run(agent, "q", config=config)
        assert result["result"] == "mock: q"

    def test_run_with_hooks(self):
        agent = MockAgent()
        started = []
        ended = []

        def on_start(q, kw):
            started.append((q, kw))

        def on_end(r):
            ended.append(r)

        config = RunConfig(hooks=RunHooks(on_start=on_start, on_end=on_end))
        result = Runner.run(agent, "test", config=config)
        assert len(started) == 1
        assert started[0][0] == "test"
        assert len(ended) == 1
        assert ended[0]["result"] == "mock: test"

    def test_run_with_real_agent(self, stategraph_agent):
        def route(state, current):
            return "end" if current == "start" else "end"

        stategraph_agent._route = route
        result = Runner.run(stategraph_agent, "hello")
        assert "result" in result
        assert "metadata" in result


class TestRunConfig:
    """Tests for RunConfig."""

    def test_default_config(self):
        config = RunConfig()
        assert config.session is None
        assert config.hooks is None
        assert config.max_retries == 0
        assert config.context == {}

    def test_config_with_values(self):
        config = RunConfig(max_retries=2, context={"x": 1})
        assert config.max_retries == 2
        assert config.context["x"] == 1
