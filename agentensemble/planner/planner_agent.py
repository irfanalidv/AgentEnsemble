"""
Planner Agent

Decomposes complex tasks into subtasks for executor agents.
"""

import asyncio
from typing import Any, Dict, List, Optional

from agentensemble.agents.base import BaseAgent
from agentensemble.core.protocol import AgentProtocol


class PlannerAgent(BaseAgent):
    """
    Planner that decomposes a task into subtasks and delegates to an executor.

    Flow: plan(task) -> [subtask1, subtask2, ...] -> executor runs each -> aggregate.
    """

    def __init__(
        self,
        name: str = "planner",
        executor: Optional[AgentProtocol] = None,
        max_subtasks: int = 10,
        **kwargs: Any,
    ):
        super().__init__(name, tools=[], max_iterations=1, **kwargs)
        self.executor = executor
        self.max_subtasks = max_subtasks

    def run(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        return asyncio.run(self.arun(query, **kwargs))

    async def arun(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        subtasks = self._decompose(query)
        if not self.executor:
            return {
                "result": f"Planned {len(subtasks)} subtasks (no executor): " + "; ".join(subtasks[:3]),
                "metadata": {"subtasks": subtasks, "executed": 0},
            }

        results: List[Dict[str, Any]] = []
        for i, st in enumerate(subtasks[: self.max_subtasks]):
            r = await self.executor.arun(st, **kwargs)
            results.append({"subtask": st, "result": r.get("result", "")})

        aggregated = self._aggregate(results)
        return {
            "result": aggregated,
            "metadata": {
                "subtasks": subtasks[: self.max_subtasks],
                "results_count": len(results),
                "agent": self.name,
            },
        }

    def _decompose(self, query: str) -> List[str]:
        """Decompose task into subtasks. Override for LLM-based planning."""
        # Simple rule-based decomposition
        q = query.lower()
        if "research" in q or "find" in q:
            return [
                f"Search for information about: {query}",
                f"Summarize key findings from: {query}",
            ]
        if "compare" in q:
            return [
                f"Analyze first option in: {query}",
                f"Analyze second option in: {query}",
                f"Compare and conclude: {query}",
            ]
        return [query]

    def _aggregate(self, results: List[Dict[str, Any]]) -> str:
        """Aggregate subtask results. Override for custom logic."""
        parts = [r.get("result", "") for r in results if r.get("result")]
        return "\n\n---\n\n".join(parts) if parts else "No results"
