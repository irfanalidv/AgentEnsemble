"""
Pipeline Orchestrator Example

Sequential agent execution - results flow from one agent to the next.
Use case: Search -> Process -> Summarize workflows.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from agentensemble.agents import ReActAgent
from agentensemble.tools import SearchTool
from agentensemble.orchestration import PipelineOrchestrator


async def main():
    searcher = ReActAgent(
        name="searcher",
        tools=[SearchTool()],
        max_iterations=2,
    )
    summarizer = ReActAgent(
        name="summarizer",
        tools=[],  # No tools - just summarization
        max_iterations=1,
    )

    pipeline = PipelineOrchestrator(
        agents={"searcher": searcher, "summarizer": summarizer},
    )

    # Sequential: searcher runs first, summarizer gets its result
    print("Running pipeline (search -> summarize)...")
    result = await pipeline.aperform(
        task="What are the key trends in AI in 2024?",
        data={"query": "AI trends 2024"},
    )

    print(f"Agents used: {result['agents_used']}")
    for name, r in result["results"].items():
        print(f"\n{name}: {r['result'][:200]}...")
    print(f"\nFinal result: {result['final_result'].get('previous_result', '')[:200]}...")


if __name__ == "__main__":
    print("\n🎭 Pipeline Orchestrator Example\n")
    asyncio.run(main())
    print("\n✅ Done!")
