"""
StateGraph Agent Example

Custom graph-based workflow with nodes and routing.
Use case: Multi-step analysis pipelines.
"""

from agentensemble.agents import StateGraphAgent, AgentState


def analyze_node(state: AgentState) -> dict:
    """Analyze the query and prepare context."""
    return {
        "context": {
            **state.context,
            "analyzed": True,
            "query_length": len(state.query),
        },
    }


def process_node(state: AgentState) -> dict:
    """Process based on analysis."""
    q = state.query.lower()
    if "python" in q:
        result = "Python is a high-level programming language."
    elif "ai" in q or "agent" in q:
        result = "AI agents are autonomous systems that use LLMs and tools."
    else:
        result = f"Processed query: {state.query[:50]}..."
    return {"result": result}


def main():
    nodes = {
        "start": lambda s: {"context": {**s.context, "started": True}},
        "analyze": analyze_node,
        "process": process_node,
    }

    agent = StateGraphAgent(
        name="analysis_agent",
        nodes=nodes,
        max_iterations=10,
    )

    # Override routing: start -> analyze -> process -> end
    original_route = agent._route

    def custom_route(state, current):
        if state.result:
            return "end"
        if current == "start":
            return "analyze"
        if current == "analyze":
            return "process"
        return "end"

    agent._route = custom_route

    result = agent.run("What is Python?")
    print("Query: What is Python?")
    print(f"Result: {result['result']}")
    print(f"Metadata: {result['metadata']}")


if __name__ == "__main__":
    print("\n🎭 StateGraph Agent Example\n")
    main()
    print("\n✅ Done!")
