"""
Reference Agent Implementations

Provides production-ready agent architectures:
- ReActAgent: Simple reasoning + acting pattern
- StateGraphAgent: Custom nodes with intelligent routing
- RAGAgent: RAG-enhanced with fallback strategies
- HybridAgent: Advanced iterative refinement
- StructuredAgent: Structured output support (langchain>=1.1 or with_structured_output)
"""

from agentensemble.agents.base import AgentState, BaseAgent
from agentensemble.agents.hybrid_agent import HybridAgent
from agentensemble.agents.rag_agent import RAGAgent
from agentensemble.agents.react_agent import ReActAgent
from agentensemble.agents.stategraph_agent import StateGraphAgent

try:
    from agentensemble.agents.structured_agent import StructuredAgent

    STRUCTURED_AVAILABLE = True
except ImportError:
    STRUCTURED_AVAILABLE = False
    StructuredAgent = None  # type: ignore[assignment,misc]

__all__ = [
    "BaseAgent",
    "AgentState",
    "ReActAgent",
    "StateGraphAgent",
    "RAGAgent",
    "HybridAgent",
]

if STRUCTURED_AVAILABLE:
    __all__.append("StructuredAgent")
