"""
Pipeline Orchestrator

Sequential agent workflow pattern.
"""

from typing import Any, Dict
from agentensemble.orchestration.ensemble import Ensemble


class PipelineOrchestrator(Ensemble):
    """
    Pipeline orchestrator - sequential workflow pattern.
    
    Agents execute in a defined sequence, passing results to the next agent.
    """
    
    def __init__(self, agents: Dict[str, Any], **kwargs):
        """
        Initialize pipeline orchestrator.
        
        Args:
            agents: Dictionary of agent name -> agent instance (order matters)
            **kwargs: Additional configuration
        """
        super().__init__(agents, conductor="pipeline", **kwargs)

