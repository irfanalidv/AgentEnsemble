"""
Workflow Graph

Composable graph-based workflows with nodes and conditional edges.
"""

from collections.abc import Callable
from typing import Any

Node = str | Callable[[dict[str, Any]], dict[str, Any]]


class Edge:
    """Edge between nodes. source -> target, optionally conditional."""

    def __init__(
        self,
        source: str,
        target: str,
        condition: Callable[[dict[str, Any]], bool] | None = None,
    ):
        self.source = source
        self.target = target
        self.condition = condition


class WorkflowGraph:
    """
    Graph-based workflow with nodes and edges.

    Simpler than LangGraph; supports sync node functions and conditional routing.
    """

    def __init__(
        self,
        nodes: dict[str, Node] | None = None,
        edges: list[Edge] | None = None,
        entry: str = "start",
        exit_nodes: list[str] | None = None,
    ):
        self.nodes = nodes or {}
        self.edges = edges or []
        self.entry = entry
        self.exit_nodes = set(exit_nodes or ["end"])

    def add_node(self, name: str, node: Node) -> "WorkflowGraph":
        """Add a node. Chainable."""
        self.nodes[name] = node
        return self

    def add_edge(
        self, source: str, target: str, condition: Callable | None = None
    ) -> "WorkflowGraph":
        """Add an edge. Chainable."""
        self.edges.append(Edge(source, target, condition))
        return self

    def run(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute the workflow."""
        state: dict[str, Any] = {
            "query": query,
            "context": context or {},
            "result": None,
            "current_node": self.entry,
        }

        for _ in range(50):  # Max steps
            current = state["current_node"]
            if current in self.exit_nodes:
                break

            node = self.nodes.get(current)
            if not node:
                break

            if callable(node):
                out = node(state)
            else:
                out = {"current_node": node}

            state.update(out)
            next_node = state.get("current_node")

            # Resolve next via edges
            for edge in self.edges:
                if edge.source == current and (not edge.condition or edge.condition(state)):
                    next_node = edge.target
                    break
            state["current_node"] = next_node or "end"

        return {
            "result": state.get("result", "No result"),
            "metadata": {"nodes_visited": list(self.nodes.keys()), "state": state},
        }
