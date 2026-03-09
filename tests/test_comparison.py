"""Tests for AgentComparison and Benchmark."""

import pytest

from agentensemble.testing import AgentComparison, Benchmark, Metrics
from agentensemble.agents import StateGraphAgent, RAGAgent


class TestAgentComparison:
    """Tests for AgentComparison."""

    def test_comparison_runs_all_agents(self, stategraph_agent):
        def route(state, current):
            return "end" if current == "start" else "end"

        stategraph_agent._route = route
        rag = RAGAgent(name="rag", tools=[])

        benchmark = Benchmark([{"query": "q1"}, {"query": "q2"}])
        comp = AgentComparison([stategraph_agent, rag])
        out = comp.run(benchmark)

        assert "results" in out
        assert "summary" in out
        assert stategraph_agent.name in out["results"]
        assert "rag" in out["results"]
        assert len(out["results"][stategraph_agent.name]) == 2
        assert len(out["results"]["rag"]) == 2

    def test_comparison_summary_has_metrics(self, stategraph_agent):
        def route(state, current):
            return "end" if current == "start" else "end"

        stategraph_agent._route = route
        benchmark = Benchmark([{"query": "test"}])
        comp = AgentComparison([stategraph_agent])
        out = comp.run(benchmark)

        summary = out["summary"][stategraph_agent.name]
        assert "total_tests" in summary
        assert "success_rate" in summary
        assert "avg_execution_time" in summary
        assert summary["total_tests"] == 1

    def test_comparison_tracks_execution_time(self, stategraph_agent):
        def route(state, current):
            return "end" if current == "start" else "end"

        stategraph_agent._route = route
        benchmark = Benchmark([{"query": "x"}])
        comp = AgentComparison([stategraph_agent])
        out = comp.run(benchmark)

        results = out["results"][stategraph_agent.name]
        assert all("execution_time" in r for r in results)
        assert all(r["execution_time"] >= 0 for r in results)


class TestBenchmark:
    """Tests for Benchmark."""

    def test_research_benchmark(self):
        b = Benchmark.research_tasks()
        assert len(b.test_cases) > 0
        assert all("query" in tc for tc in b.test_cases)

    def test_extraction_benchmark(self):
        b = Benchmark.data_extraction_tasks()
        assert len(b.test_cases) > 0

    def test_load_by_name(self):
        b = Benchmark.load("research_tasks")
        assert b is not None
        assert len(b.test_cases) > 0

    def test_load_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown benchmark"):
            Benchmark.load("unknown")


class TestMetrics:
    """Tests for Metrics."""

    def test_success_rate_empty(self):
        assert Metrics.success_rate([]) == 0.0

    def test_success_rate_all_success(self):
        results = [{"success": True}, {"success": True}]
        assert Metrics.success_rate(results) == 1.0

    def test_success_rate_mixed(self):
        results = [{"success": True}, {"success": False}]
        assert Metrics.success_rate(results) == 0.5

    def test_average_execution_time(self):
        results = [{"execution_time": 1.0}, {"execution_time": 3.0}]
        assert Metrics.average_execution_time(results) == 2.0

    def test_calculate_all(self):
        results = [
            {"success": True, "execution_time": 1.0, "cost": 0},
            {"success": False, "execution_time": 2.0, "cost": 0},
        ]
        m = Metrics.calculate_all(results)
        assert m["total_tests"] == 2
        assert m["success_rate"] == 0.5
        assert m["average_execution_time"] == 1.5
