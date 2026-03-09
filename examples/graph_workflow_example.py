"""
Graph Workflow Example

WorkflowGraph for composable, graph-based workflows with conditional edges.
"""

from agentensemble import WorkflowGraph, Edge


def analyze_node(state):
    """Analyze the query."""
    return {"context": {**state.get("context", {}), "analyzed": True}}


def search_node(state):
    """Simulate search step."""
    q = state.get("query", "")
    ctx = state.get("context", {})
    return {"context": {**ctx, "search_result": f"Results for: {q}"}}


def summarize_node(state):
    """Produce final summary."""
    ctx = state.get("context", {})
    return {
        "result": f"Summary: {ctx.get('search_result', 'N/A')}",
        "current_node": "end",
    }


def main():
    graph = (
        WorkflowGraph(entry="start", exit_nodes=["end"])
        .add_node("start", lambda s: {"current_node": "analyze", "context": s.get("context", {})})
        .add_node("analyze", analyze_node)
        .add_node("search", search_node)
        .add_node("summarize", summarize_node)
        .add_edge("start", "analyze")
        .add_edge("analyze", "search")
        .add_edge("search", "summarize")
        .add_edge("summarize", "end")
    )

    # Run workflow
    result = graph.run("What is multi-agent AI?")
    print("Result:", result["result"])
    print("Metadata:", result["metadata"].get("nodes_visited", []))


if __name__ == "__main__":
    main()
