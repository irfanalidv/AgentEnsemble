"""
Agent Comparison

Compare multiple agent implementations on the same tasks with real metrics.
"""

import time
from typing import Any

from agentensemble.agents.base import BaseAgent
from agentensemble.testing.metrics import Metrics


def _is_successful_result(result: dict[str, Any]) -> bool:
    """Check if agent result is considered successful."""
    if not result:
        return False
    r = result.get("result", "")
    if not r or not str(r).strip():
        return False
    s = str(r).lower().strip()
    if s in (
        "no result",
        "no result generated",
        "not found",
        "no result found",
        "unable to generate answer",
    ):
        return False
    if len(s) < 5:
        return False
    return True


class AgentComparison:
    """
    Compare multiple agent implementations.

    Runs the same tasks on different agents and computes real metrics:
    success_rate, avg_execution_time, total_tests.
    """

    def __init__(self, agents: list[BaseAgent]):
        """
        Initialize agent comparison.

        Args:
            agents: List of agent instances to compare
        """
        self.agents = agents

    def run(
        self,
        benchmark: Any,
        metrics: list[str] | None = None,
        use_async: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Run comparison benchmark.

        Args:
            benchmark: Benchmark instance with test_cases
            metrics: List of metrics to track (default: success_rate, execution_time)
            use_async: Use arun() instead of run() for async agents
            **kwargs: Passed to agent.run() or agent.arun()

        Returns:
            Comparison results with real computed metrics
        """
        import asyncio

        metrics_list = metrics or ["success_rate", "execution_time", "total_tests"]
        results: dict[str, list[dict[str, Any]]] = {}

        for agent in self.agents:
            agent_results = []

            for test_case in benchmark.test_cases:
                query = test_case.get("query", "")
                start = time.perf_counter()
                try:
                    if use_async and hasattr(agent, "arun"):
                        result = asyncio.run(agent.arun(query, **kwargs))
                    else:
                        result = agent.run(query, **kwargs)
                except Exception as e:
                    result = {"result": f"Error: {e}", "metadata": {"error": str(e)}}
                elapsed = time.perf_counter() - start

                agent_results.append(
                    {
                        "test_case": test_case,
                        "result": result,
                        "execution_time": round(elapsed, 3),
                        "success": _is_successful_result(result),
                    }
                )

            results[agent.name] = agent_results

        return {
            "results": results,
            "metrics": metrics_list,
            "summary": self._calculate_summary(results, metrics_list),
        }

    def _calculate_summary(
        self,
        results: dict[str, list[dict[str, Any]]],
        metrics_list: list[str],
    ) -> dict[str, Any]:
        """Calculate summary statistics using Metrics."""
        summary: dict[str, Any] = {}

        for agent_name, agent_results in results.items():
            m = Metrics.calculate_all(agent_results)
            summary[agent_name] = {
                "total_tests": m["total_tests"],
                "success_rate": round(m["success_rate"], 2),
                "avg_execution_time": round(m["average_execution_time"], 2),
                "total_execution_time": round(sum(r["execution_time"] for r in agent_results), 2),
            }

        return summary
