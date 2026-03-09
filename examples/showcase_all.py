"""
AgentEnsemble Full Showcase

Demonstrates ALL framework capabilities with MISTRAL_API_KEY and SERPER_API_KEY from .env.
Real-world examples with actual API calls.

Setup:
    Create .env in project root (no spaces around =):
        MISTRAL_API_KEY=your-mistral-key
        SERPER_API_KEY=your-serper-key

Run:
    PYTHONPATH=. python examples/showcase_all.py
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

# Validate keys before running
MISTRAL = os.getenv("MISTRAL_API_KEY", "").strip()
SERPER = os.getenv("SERPER_API_KEY", "").strip()
if not MISTRAL:
    print("⚠️  MISTRAL_API_KEY not found in .env - some examples will use fallbacks")
if not SERPER:
    print("⚠️  SERPER_API_KEY not found in .env - SearchTool will use DuckDuckGo fallback")

from agentensemble import (
    ReActAgent,
    HybridAgent,
    StateGraphAgent,
    Ensemble,
    RouterAgent,
    PlannerAgent,
    WorkflowGraph,
    Runner,
    RunConfig,
    RunHooks,
    SearchTool,
    InMemorySession,
)
from agentensemble.orchestration import SwarmOrchestrator, PipelineOrchestrator, DebateOrchestrator
from agentensemble.tools import function_tool


# --- 1. ReActAgent: LLM + Search (real web search) ---
async def demo_react_search():
    print("\n" + "=" * 70)
    print("1. ReActAgent — LLM decides when to search vs answer directly")
    print("=" * 70)

    agent = ReActAgent(name="research", tools=[SearchTool()], max_iterations=5)
    result = await agent.arun("Who won the 2024 Nobel Prize in Physics?")
    print(f"Query: Who won the 2024 Nobel Prize in Physics?")
    print(f"Result: {result['result'][:500]}...")
    print(f"Metadata: {result['metadata']}")


# --- 2. Custom @function_tool ---
@function_tool(description="Get current weather for a city")
def get_weather(city: str) -> str:
    return f"Weather in {city}: 72°F, partly cloudy"


async def demo_custom_tool():
    print("\n" + "=" * 70)
    print("2. Custom @function_tool — Weather + Search")
    print("=" * 70)

    agent = ReActAgent(name="assistant", tools=[SearchTool(), get_weather], max_iterations=4)
    result = await agent.arun("What is the weather in Paris? Use the weather tool.")
    print(f"Query: What is the weather in Paris?")
    print(f"Result: {result['result'][:400]}...")
    print(f"Tool calls: {result['metadata'].get('tool_calls', 0)}")


# --- 3. HybridAgent — LLM routing (search → RAG → validate → answer) ---
async def demo_hybrid():
    print("\n" + "=" * 70)
    print("3. HybridAgent — LLM-based routing (SEARCH → RAG → VALIDATE → ANSWER)")
    print("=" * 70)

    agent = HybridAgent(name="hybrid", tools=[SearchTool()], max_iterations=5)
    result = await agent.arun("What are the key trends in AI agents in 2024?")
    print(f"Query: What are the key trends in AI agents in 2024?")
    print(f"Result: {result['result'][:500]}...")
    print(f"Actions taken: {result['metadata'].get('actions_taken', [])}")


# --- 4. Swarm — Parallel agents ---
async def demo_swarm():
    print("\n" + "=" * 70)
    print("4. Swarm — Parallel agents (asyncio.gather)")
    print("=" * 70)

    researcher = ReActAgent(name="researcher", tools=[SearchTool()], max_iterations=2)
    fact_checker = ReActAgent(name="fact_checker", tools=[SearchTool()], max_iterations=2)

    swarm = SwarmOrchestrator(agents={"researcher": researcher, "fact_checker": fact_checker})
    result = await swarm.aperform(
        task="What is the capital of France?",
        data={"query": "capital of France"},
    )
    print(f"Task: What is the capital of France?")
    print(f"Agents used: {result['agents_used']}")
    for name, r in result["results"].items():
        print(f"  {name}: {r['result'][:120]}...")


# --- 5. Pipeline — Sequential (search → summarize) ---
async def demo_pipeline():
    print("\n" + "=" * 70)
    print("5. Pipeline — Sequential workflow (search → summarize)")
    print("=" * 70)

    searcher = ReActAgent(name="searcher", tools=[SearchTool()], max_iterations=2)
    summarizer = ReActAgent(name="summarizer", tools=[], max_iterations=1)

    pipeline = PipelineOrchestrator(agents={"searcher": searcher, "summarizer": summarizer})
    result = await pipeline.aperform(
        task="Summarize the top 3 AI breakthroughs in 2024 in 2 sentences each",
        data={"query": "AI breakthroughs 2024"},
    )
    print(f"Task: Summarize top 3 AI breakthroughs in 2024")
    print(f"Agents used: {result['agents_used']}")
    final = result.get("final_result", {}).get("previous_result", "")
    print(f"Final: {final[:400]}...")


# --- 6. Debate — Multi-agent consensus ---
async def demo_debate():
    print("\n" + "=" * 70)
    print("6. Debate — Solvers propose → exchange feedback → aggregator votes")
    print("=" * 70)

    solver1 = ReActAgent(name="solver_1", tools=[], max_iterations=1)
    solver2 = ReActAgent(name="solver_2", tools=[], max_iterations=1)
    aggregator = ReActAgent(name="aggregator", tools=[], max_iterations=1)

    debate = DebateOrchestrator(solvers=[solver1, solver2], aggregator=aggregator, rounds=2)
    result = await debate.adebate("What is 15 * 7?")
    print(f"Problem: What is 15 * 7?")
    print(f"Proposals: {result.get('proposals', [])}")
    print(f"Final answer: {result['result']}")


# --- 7. Router + Ensemble — LLM-based agent selection ---
async def demo_router():
    print("\n" + "=" * 70)
    print("7. Router + Ensemble — LLM routes to best agent")
    print("=" * 70)

    researcher = ReActAgent(name="researcher", tools=[SearchTool()], max_iterations=2)
    validator = ReActAgent(name="validator", tools=[SearchTool()], max_iterations=2)

    router = RouterAgent(agents={"researcher": researcher, "validator": validator})
    ensemble = Ensemble(
        conductor="supervisor",
        agents={"researcher": researcher, "validator": validator},
        router=router,
    )
    result = await ensemble.aperform("What are the latest developments in quantum computing?")
    print(f"Task: Latest developments in quantum computing")
    for name, r in result.get("results", {}).items():
        print(f"  Routed to {name}: {r.get('result', '')[:150]}...")
        if r.get("metadata", {}).get("routed_to"):
            print(f"  (Router selected: {r['metadata']['routed_to']})")


# --- 8. Planner — Task decomposition ---
async def demo_planner():
    print("\n" + "=" * 70)
    print("8. Planner — Decompose task → executor runs subtasks")
    print("=" * 70)

    def executor_node(state):
        return {"result": f"Processed: {state.query}"}

    executor = StateGraphAgent(name="worker", nodes={"start": executor_node})
    executor._route = lambda s, c: "end"

    planner = PlannerAgent(executor=executor, max_subtasks=3)
    result = await planner.arun("Research and compare LangGraph vs CrewAI")
    print(f"Task: Research and compare LangGraph vs CrewAI")
    print(f"Subtasks: {result['metadata'].get('subtasks', [])}")
    print(f"Result: {result['result'][:300]}...")


# --- 9. WorkflowGraph — Custom DAG ---
def demo_graph():
    print("\n" + "=" * 70)
    print("9. WorkflowGraph — Composable graph with nodes and edges")
    print("=" * 70)

    def analyze(s):
        return {"context": {**s.get("context", {}), "analyzed": True}, "current_node": "search"}

    def search(s):
        q = s.get("query", "")
        return {"context": {**s.get("context", {}), "found": f"Results for: {q}"}, "current_node": "summarize"}

    def summarize(s):
        ctx = s.get("context", {})
        return {"result": f"Summary: {ctx.get('found', 'N/A')}", "current_node": "end"}

    graph = (
        WorkflowGraph(entry="start", exit_nodes=["end"])
        .add_node("start", lambda s: {"current_node": "analyze"})
        .add_node("analyze", analyze)
        .add_node("search", search)
        .add_node("summarize", summarize)
        .add_edge("start", "analyze")
        .add_edge("analyze", "search")
        .add_edge("search", "summarize")
        .add_edge("summarize", "end")
    )
    result = graph.run("What is multi-agent AI?")
    print(f"Query: What is multi-agent AI?")
    print(f"Result: {result['result']}")


# --- 10. Runner + Hooks + Memory ---
async def demo_runner_memory():
    print("\n" + "=" * 70)
    print("10. Runner + RunHooks + InMemorySession")
    print("=" * 70)

    session = InMemorySession(session_id="showcase-1")
    agent = ReActAgent(
        name="chatbot",
        tools=[SearchTool()],
        session=session,
        max_iterations=3,
    )

    events = []

    def on_start(query, kwargs):
        events.append(f"START: {query[:40]}...")

    def on_end(result):
        events.append(f"END: {str(result.get('result', ''))[:40]}...")

    config = RunConfig(hooks=RunHooks(on_start=on_start, on_end=on_end))
    result = await Runner.arun(agent, "What is the capital of Japan?", config=config)
    print(f"User: What is the capital of Japan?")
    print(f"Agent: {result['result'][:200]}...")
    print(f"Hooks fired: {events}")
    print(f"Session messages: {len(session.get_messages())}")

    # Turn 2 - agent remembers context
    result2 = await Runner.arun(agent, "What about its population?", config=config)
    print(f"\nUser: What about its population?")
    print(f"Agent: {result2['result'][:200]}...")
    print(f"Session messages: {len(session.get_messages())}")


# --- Main ---
async def main():
    print("\n" + "🎭" * 35)
    print("   AgentEnsemble Full Showcase — All Features")
    print("🎭" * 35)

    async_demos = [
        demo_react_search,
        demo_custom_tool,
        demo_hybrid,
        demo_swarm,
        demo_pipeline,
        demo_debate,
        demo_router,
        demo_planner,
        demo_runner_memory,
    ]
    for fn in async_demos:
        try:
            await fn()
        except Exception as e:
            print(f"\n⚠️  {fn.__name__} failed: {e}")

    try:
        demo_graph()
    except Exception as e:
        print(f"\n⚠️  demo_graph failed: {e}")

    print("\n" + "=" * 70)
    print("✅ Showcase complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
