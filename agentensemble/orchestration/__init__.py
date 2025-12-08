"""
Orchestration Patterns

Provides different patterns for coordinating multiple agents:
- Ensemble: Full multi-agent coordination
- Supervisor: Central coordinator pattern
- Swarm: Decentralized collaboration
- Pipeline: Sequential workflows
"""

from agentensemble.orchestration.ensemble import Ensemble
from agentensemble.orchestration.supervisor import SupervisorOrchestrator
from agentensemble.orchestration.swarm import SwarmOrchestrator
from agentensemble.orchestration.pipeline import PipelineOrchestrator

__all__ = [
    "Ensemble",
    "SupervisorOrchestrator",
    "SwarmOrchestrator",
    "PipelineOrchestrator",
]

