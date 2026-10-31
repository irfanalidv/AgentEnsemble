"""
Supervisor Orchestrator

Central coordinator pattern for managing specialized agents.
"""

from typing import Any

from agentensemble.orchestration.ensemble import Ensemble


class SupervisorOrchestrator(Ensemble):
    """
    Supervisor orchestrator - central coordinator pattern.

    A supervisor agent manages and delegates tasks to specialized agents.
    """

    def __init__(self, agents: dict[str, Any], **kwargs):
        """
        Initialize supervisor orchestrator.

        Args:
            agents: Dictionary of agent name -> agent instance
            **kwargs: Additional configuration
        """
        super().__init__(agents, conductor="supervisor", **kwargs)
