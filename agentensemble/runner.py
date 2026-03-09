"""
Runner - Central Entry Point

Unified execution with RunConfig, hooks, and error handling.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Protocol, runtime_checkable

from agentensemble.memory import Session


@runtime_checkable
class AgentProtocol(Protocol):
    """Protocol for runnable agents."""

    name: str

    def run(self, query: str, **kwargs: Any) -> Dict[str, Any]: ...
    async def arun(self, query: str, **kwargs: Any) -> Dict[str, Any]: ...


@dataclass
class RunHooks:
    """Lifecycle hooks for run execution."""

    on_start: Optional[Callable[[str, Dict[str, Any]], None]] = None
    on_end: Optional[Callable[[Dict[str, Any]], None]] = None
    on_error: Optional[Callable[[Exception], None]] = None


@dataclass
class RunConfig:
    """Configuration for Runner execution."""

    session: Optional[Session] = None
    hooks: Optional[RunHooks] = None
    max_retries: int = 0
    retry_on: tuple = (Exception,)  # Exception types to retry (e.g. (RateLimitError, TimeoutError))
    context: Dict[str, Any] = field(default_factory=dict)


class Runner:
    """
    Central entry point for agent execution.

    Provides unified run/arun with config, hooks, and error handling.
    """

    @staticmethod
    def run(
        agent: AgentProtocol,
        input: str,
        config: Optional[RunConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute agent (sync).

        Args:
            agent: Agent implementing run() and arun()
            input: Query string
            config: Optional RunConfig (session, hooks, etc.)
            **kwargs: Additional kwargs passed to agent

        Returns:
            Result dict from agent
        """
        return asyncio.run(Runner.arun(agent, input, config, **kwargs))

    @staticmethod
    async def arun(
        agent: AgentProtocol,
        input: str,
        config: Optional[RunConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute agent (async).

        Args:
            agent: Agent implementing run() and arun()
            input: Query string
            config: Optional RunConfig
            **kwargs: Additional kwargs passed to agent

        Returns:
            Result dict from agent
        """
        config = config or RunConfig()
        merged_kwargs = {**config.context, **kwargs}
        if config.session and hasattr(agent, "session"):
            merged_kwargs.setdefault("session", config.session)
        elif config.session:
            agent = _inject_session(agent, config.session)

        if config.hooks and config.hooks.on_start:
            config.hooks.on_start(input, merged_kwargs)

        last_error: Optional[Exception] = None
        for attempt in range(config.max_retries + 1):
            try:
                result = await agent.arun(input, **merged_kwargs)
                if config.hooks and config.hooks.on_end:
                    config.hooks.on_end(result)
                return result
            except Exception as e:
                last_error = e
                if config.hooks and config.hooks.on_error:
                    config.hooks.on_error(e)
                if attempt == config.max_retries:
                    raise
                retryable = any(isinstance(e, t) for t in config.retry_on)
                if not retryable:
                    raise

        raise last_error or RuntimeError("Unexpected error")


def _inject_session(agent: Any, session: Session) -> Any:
    """Inject session into agent if it supports it."""
    if hasattr(agent, "session"):
        agent.session = session
    return agent
