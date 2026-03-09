"""
Debate Orchestrator Example

Multi-agent debate: solvers propose, exchange feedback, aggregator votes.
Use case: Math problems, reasoning tasks where consensus improves accuracy.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from agentensemble.agents import ReActAgent
from agentensemble.orchestration import DebateOrchestrator


async def main():
    # Solvers (no tools needed for simple math)
    solver1 = ReActAgent(name="solver_1", tools=[], max_iterations=1)
    solver2 = ReActAgent(name="solver_2", tools=[], max_iterations=1)
    aggregator = ReActAgent(name="aggregator", tools=[], max_iterations=1)

    debate = DebateOrchestrator(
        solvers=[solver1, solver2],
        aggregator=aggregator,
        rounds=2,
    )

    print("Running debate on: What is 15 * 7?")
    result = await debate.adebate("What is 15 * 7?")
    print(f"\nProposals: {result['proposals']}")
    print(f"Final answer: {result['result']}")
    print(f"Metadata: {result['metadata']}")


if __name__ == "__main__":
    print("\n🎭 Debate Orchestrator Example\n")
    asyncio.run(main())
    print("\n✅ Done!")
