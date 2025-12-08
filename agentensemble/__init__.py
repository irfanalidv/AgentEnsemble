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
)

from agentensemble.tools import (
    ToolRegistry,
    SearchTool,
    ScraperTool,
    RAGTool,
    ValidationTool,
)

from agentensemble.testing import (
    AgentComparison,
    Benchmark,
    Metrics,
)

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
    # Tools
    "ToolRegistry",
    "SearchTool",
    "ScraperTool",
    "RAGTool",
    "ValidationTool",
    # Testing
    "AgentComparison",
    "Benchmark",
    "Metrics",
]

# Add StructuredAgent if available
if STRUCTURED_AVAILABLE:
    __all__.append("StructuredAgent")

