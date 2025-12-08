"""
Swarm Orchestrator

Decentralized agent collaboration pattern.
"""

from typing import Any, Dict
from agentensemble.orchestration.ensemble import Ensemble


class SwarmOrchestrator(Ensemble):
    """
    Swarm orchestrator - decentralized collaboration pattern.
    
    Agents operate independently and collaborate organically.
    """
    
    def __init__(self, agents: Dict[str, Any], **kwargs):
        """
        Initialize swarm orchestrator.
        
        Args:
            agents: Dictionary of agent name -> agent instance
            **kwargs: Additional configuration
        """
        super().__init__(agents, conductor="swarm", **kwargs)

