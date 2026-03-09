"""
Unified Agent Protocol

All agents implement AgentProtocol for consistent API and interoperability.
Enables Runner, Ensemble, and orchestration to work with any agent type.
"""

from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class AgentProtocol(Protocol):
    """
    Protocol for all agents in AgentEnsemble.

    Implement run() and arun() for sync/async execution.
    Return shape: {"result": str, "metadata": dict}
    """

    name: str

    def run(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        """Execute agent (sync). Returns {result, metadata}."""
        ...

    async def arun(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        """Execute agent (async). Returns {result, metadata}."""
        ...


@runtime_checkable
class RunnableProtocol(Protocol):
    """Minimal protocol for runnable objects (agents, tools)."""

    def run(self, query: str, **kwargs: Any) -> Any:
        ...
