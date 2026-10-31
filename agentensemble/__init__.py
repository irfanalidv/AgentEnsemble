"""
AgentEnsemble - Orchestrate AI agents in perfect harmony

A comprehensive framework for building, testing, and orchestrating multi-agent AI systems.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("agentensemble")
except PackageNotFoundError:  # running from a source checkout without installation
    __version__ = "0.5.0"
__author__ = "Irfan Ali"

# Core imports
from agentensemble.agents import (
    HybridAgent,
    RAGAgent,
    ReActAgent,
    StateGraphAgent,
)

# Structured output agent (optional, requires langchain>=1.1)
try:
    from agentensemble.agents import StructuredAgent

    STRUCTURED_AVAILABLE = True
except ImportError:
    STRUCTURED_AVAILABLE = False
    StructuredAgent = None  # type: ignore[assignment,misc]

from agentensemble.core import AgentProtocol
from agentensemble.graph import Edge, Node, WorkflowGraph
from agentensemble.memory import InMemorySession, Session, SQLiteSession
from agentensemble.orchestration import (
    DebateOrchestrator,
    Ensemble,
    PipelineOrchestrator,
    SupervisorOrchestrator,
    SwarmOrchestrator,
)
from agentensemble.planner import PlannerAgent
from agentensemble.router import RouterAgent
from agentensemble.runner import RunConfig, RunHooks, Runner
from agentensemble.testing import (
    AgentComparison,
    Benchmark,
    Metrics,
)
from agentensemble.tools import (
    FunctionTool,
    RAGTool,
    ScraperTool,
    SearchTool,
    ToolRegistry,
    ValidationTool,
    function_tool,
)
from agentensemble.tracing import TraceEvent, TraceEventType, TraceHooks, estimate_cost

__all__ = [
    # Version
    "__version__",
    "__author__",
    # Agents
    "ReActAgent",
    "StateGraphAgent",
    "RAGAgent",
    "HybridAgent",
    # Orchestration
    "Ensemble",
    "SupervisorOrchestrator",
    "SwarmOrchestrator",
    "PipelineOrchestrator",
    "DebateOrchestrator",
    # Memory
    "Session",
    "InMemorySession",
    "SQLiteSession",
    # Tools
    "ToolRegistry",
    "SearchTool",
    "ScraperTool",
    "RAGTool",
    "ValidationTool",
    "FunctionTool",
    "function_tool",
    # Testing
    "AgentComparison",
    "Benchmark",
    "Metrics",
    # Runner
    "Runner",
    "RunConfig",
    "RunHooks",
    # Core
    "AgentProtocol",
    # Router & Planner
    "RouterAgent",
    "PlannerAgent",
    # Graph
    "WorkflowGraph",
    "Node",
    "Edge",
    # Tracing
    "TraceHooks",
    "TraceEvent",
    "TraceEventType",
    "estimate_cost",
]

# Add StructuredAgent if available
if STRUCTURED_AVAILABLE:
    __all__.append("StructuredAgent")
