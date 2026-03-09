"""
AgentEnsemble - Orchestrate AI agents in perfect harmony

A comprehensive framework for building, testing, and orchestrating multi-agent AI systems.
"""

__version__ = "0.1.0"
__author__ = "Irfan Ali"

# Core imports
from agentensemble.agents import (
    ReActAgent,
    StateGraphAgent,
    RAGAgent,
    HybridAgent,
)

# Structured output agent (optional, requires langchain>=1.1)
try:
    from agentensemble.agents import StructuredAgent
    STRUCTURED_AVAILABLE = True
except ImportError:
    STRUCTURED_AVAILABLE = False
    StructuredAgent = None

from agentensemble.orchestration import (
    Ensemble,
    SupervisorOrchestrator,
    SwarmOrchestrator,
    PipelineOrchestrator,
    DebateOrchestrator,
)

from agentensemble.memory import Session, InMemorySession, SQLiteSession
from agentensemble.tools import (
    ToolRegistry,
    SearchTool,
    ScraperTool,
    RAGTool,
    ValidationTool,
    FunctionTool,
    function_tool,
)

from agentensemble.testing import (
    AgentComparison,
    Benchmark,
    Metrics,
)
from agentensemble.runner import Runner, RunConfig, RunHooks
from agentensemble.core import AgentProtocol
from agentensemble.router import RouterAgent
from agentensemble.planner import PlannerAgent
from agentensemble.graph import WorkflowGraph, Node, Edge
from agentensemble.tracing import TraceHooks, TraceEvent

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
]

# Add StructuredAgent if available
if STRUCTURED_AVAILABLE:
    __all__.append("StructuredAgent")

