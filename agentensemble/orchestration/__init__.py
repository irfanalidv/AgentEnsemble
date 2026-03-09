"""
Orchestration Patterns

Provides different patterns for coordinating multiple agents:
- Ensemble: Full multi-agent coordination (Supervisor, Swarm, Pipeline)
- DebateOrchestrator: Multi-agent debate for reasoning tasks
"""

from agentensemble.orchestration.ensemble import Ensemble
from agentensemble.orchestration.supervisor import SupervisorOrchestrator
from agentensemble.orchestration.swarm import SwarmOrchestrator
from agentensemble.orchestration.pipeline import PipelineOrchestrator
from agentensemble.orchestration.debate import DebateOrchestrator

__all__ = [
    "Ensemble",
    "SupervisorOrchestrator",
    "SwarmOrchestrator",
    "PipelineOrchestrator",
    "DebateOrchestrator",
]

