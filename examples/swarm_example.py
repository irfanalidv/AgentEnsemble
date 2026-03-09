"""
Swarm Orchestrator Example

Parallel agent execution - all agents run concurrently.
Use case: Gather multiple perspectives on the same question.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from agentensemble.agents import ReActAgent
from agentensemble.tools import SearchTool
from agentensemble.orchestration import SwarmOrchestrator


async def main():
    # Create specialized agents
    researcher = ReActAgent(
        name="researcher",
        tools=[SearchTool()],
        max_iterations=2,
    )
    fact_checker = ReActAgent(
        name="fact_checker",
        tools=[SearchTool()],
        max_iterations=2,
    )

    swarm = SwarmOrchestrator(
        agents={"researcher": researcher, "fact_checker": fact_checker},
    )

    # Run in parallel (async)
    print("Running swarm in parallel...")
    result = await swarm.aperform(
        task="What is the capital of France?",
        data={"query": "capital of France"},
    )

    print(f"Agents used: {result['agents_used']}")
    for name, r in result["results"].items():
        print(f"\n{name}: {r['result'][:150]}...")


if __name__ == "__main__":
    print("\n🎭 Swarm Orchestrator Example\n")
    asyncio.run(main())
    print("\n✅ Done!")
