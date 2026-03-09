"""
Debate Orchestrator

Multi-agent debate pattern: solvers propose, exchange feedback, aggregator votes.
Research: AutoGen, CrewAI - effective for math/reasoning (e.g., GSM8K).
"""

import asyncio
from typing import Any, Dict, List

from agentensemble.agents.base import BaseAgent


class DebateOrchestrator:
    """
    Multi-agent debate: solvers propose solutions, exchange critiques, aggregator decides.

    Flow:
    1. Aggregator distributes problem to solvers
    2. Each solver proposes initial answer
    3. Solvers exchange feedback (rounds)
    4. Aggregator votes on final answer
    """

    def __init__(
        self,
        solvers: List[BaseAgent],
        aggregator: BaseAgent,
        rounds: int = 2,
        **kwargs: Any,
    ):
        """
        Initialize debate orchestrator.

        Args:
            solvers: Agents that propose and refine solutions
            aggregator: Agent that distributes problem and votes on final answer
            rounds: Number of feedback exchange rounds
            **kwargs: Additional config
        """
        self.solvers = solvers
        self.aggregator = aggregator
        self.rounds = rounds
        self.config = kwargs

    def debate(self, problem: str, **kwargs: Any) -> Dict[str, Any]:
        """Run debate (sync)."""
        return asyncio.run(self.adebate(problem, **kwargs))

    async def adebate(self, problem: str, **kwargs: Any) -> Dict[str, Any]:
        """Run debate (async)."""
        # Round 0: All solvers propose
        tasks = [solver.arun(problem, **kwargs) for solver in self.solvers]
        initial_results = await asyncio.gather(*tasks)
        proposals = [r.get("result", "") for r in initial_results]

        # Build context for feedback rounds
        context = {
            "problem": problem,
            "proposals": proposals,
            "round": 0,
        }

        # Feedback rounds (simplified: solvers see others' proposals)
        for r in range(1, self.rounds):
            feedback_context = {
                **context,
                "round": r,
                "other_proposals": "\n\n".join(
                    f"Solver {i+1}: {p}" for i, p in enumerate(proposals)
                ),
            }
            tasks = [
                solver.arun(
                    f"Problem: {problem}\n\nOther proposals:\n{feedback_context['other_proposals']}\n\nRefine your answer.",
                    context=feedback_context,
                    **kwargs,
                )
                for solver in self.solvers
            ]
            refined = await asyncio.gather(*tasks)
            proposals = [x.get("result", "") for x in refined]
            context["proposals"] = proposals

        # Aggregator votes
        vote_prompt = (
            f"Problem: {problem}\n\n"
            + "Proposed answers:\n"
            + "\n\n".join(f"Option {i+1}: {p}" for i, p in enumerate(proposals))
            + "\n\nChoose the best answer (or synthesize one)."
        )
        final = await self.aggregator.arun(vote_prompt, **kwargs)

        return {
            "result": final.get("result", ""),
            "proposals": proposals,
            "metadata": {
                "rounds": self.rounds,
                "num_solvers": len(self.solvers),
                "aggregator": self.aggregator.name,
            },
        }
