"""
Research Agent Example

End-to-end research workflow: Planner → Router → Executor.
Uses StateGraphAgent (no API key) for demo; swap for ReActAgent + SearchTool for production.
"""

from agentensemble import (
    PlannerAgent,
    RouterAgent,
    StateGraphAgent,
    Ensemble,
    Runner,
    RunConfig,
)


def main():
    # Executor: simple graph that processes queries (state is AgentState)
    def start_node(state):
        return {"result": f"Research summary for: {state.query}"}

    executor = StateGraphAgent(
        name="researcher",
        nodes={"start": start_node},
        max_iterations=5,
    )
    executor._route = lambda s, c: "end"

    # Router: selects best agent (single agent here, so trivial)
    router = RouterAgent(name="router", agents={"researcher": executor}, llm=None)

    # Planner: decomposes task, runs executor on subtasks
    planner = PlannerAgent(name="planner", executor=executor, max_subtasks=3)

    # Option 1: Planner-only (task decomposition)
    print("=== Planner (task decomposition) ===")
    result = planner.run("Research quantum computing breakthroughs 2024")
    print(result["result"][:200], "...")
    print("Metadata:", result["metadata"])

    # Option 2: Router + Ensemble (LLM-based agent selection when router has LLM)
    print("\n=== Router + Ensemble ===")
    ensemble = Ensemble(
        conductor="supervisor",
        agents={"researcher": executor},
        router=router,
    )
    result = ensemble.perform("What are AI agent frameworks?")
    print("Result:", result["results"].get("researcher", {}).get("result", "")[:150])

    # Option 3: Runner with config
    print("\n=== Runner ===")
    result = Runner.run(executor, "Summarize agent orchestration patterns", RunConfig())
    print("Result:", result["result"][:150])


if __name__ == "__main__":
    main()
