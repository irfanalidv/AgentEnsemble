"""
Ensemble Orchestration Pattern

Coordinates multiple agents working together in harmony.
"""

from typing import Any, Dict, List, Optional
from agentensemble.agents.base import BaseAgent


class Ensemble:
    """
    Ensemble orchestrator for coordinating multiple agents.
    
    Supports different coordination modes:
    - supervisor: Central coordinator manages agents
    - swarm: Decentralized agent collaboration
    - pipeline: Sequential agent workflows
    """
    
    def __init__(
        self,
        agents: Dict[str, BaseAgent],
        conductor: str = "supervisor",
        **kwargs
    ):
        """
        Initialize ensemble.
        
        Args:
            agents: Dictionary of agent name -> agent instance
            conductor: Coordination mode ("supervisor", "swarm", "pipeline")
            **kwargs: Additional configuration
        """
        self.agents = agents
        self.conductor = conductor
        self.config = kwargs
    
    def perform(self, task: str, data: Any = None, **kwargs) -> Dict[str, Any]:
        """
        Execute ensemble of agents on a task.
        
        Args:
            task: Task description
            data: Input data
            **kwargs: Additional parameters
            
        Returns:
            Combined results from all agents
        """
        if self.conductor == "supervisor":
            return self._supervisor_coordinate(task, data, **kwargs)
        elif self.conductor == "swarm":
            return self._swarm_coordinate(task, data, **kwargs)
        elif self.conductor == "pipeline":
            return self._pipeline_coordinate(task, data, **kwargs)
        else:
            raise ValueError(f"Unknown conductor mode: {self.conductor}")
    
    def _supervisor_coordinate(
        self,
        task: str,
        data: Any,
        **kwargs
    ) -> Dict[str, Any]:
        """Supervisor pattern: Central agent coordinates others"""
        results = {}
        
        # Determine which agents to use
        agent_order = self._determine_agent_order(task)
        
        for agent_name in agent_order:
            if agent_name in self.agents:
                agent = self.agents[agent_name]
                # Ensure context is always a dict
                context = data if isinstance(data, dict) else {"query": str(data), "original_data": data}
                result = agent.run(task, context=context, **kwargs)
                results[agent_name] = result
                
                # Update context for next agent (keep as dict)
                data = {"previous_result": result.get("result", ""), "metadata": result.get("metadata", {})}
        
        return {
            "results": results,
            "conductor": "supervisor",
            "agents_used": agent_order,
        }
    
    def _swarm_coordinate(
        self,
        task: str,
        data: Any,
        **kwargs
    ) -> Dict[str, Any]:
        """Swarm pattern: Agents collaborate independently"""
        results = {}
        
        # All agents work in parallel (simplified - would use async in production)
        context = data if isinstance(data, dict) else {"query": str(data), "original_data": data}
        for agent_name, agent in self.agents.items():
            result = agent.run(task, context=context, **kwargs)
            results[agent_name] = result
        
        # Combine results
        return {
            "results": results,
            "conductor": "swarm",
            "agents_used": list(self.agents.keys()),
        }
    
    def _pipeline_coordinate(
        self,
        task: str,
        data: Any,
        **kwargs
    ) -> Dict[str, Any]:
        """Pipeline pattern: Sequential agent execution"""
        results = {}
        current_data = data
        
        # Execute agents in order
        for agent_name, agent in self.agents.items():
            # Ensure context is always a dict
            context = current_data if isinstance(current_data, dict) else {"query": str(current_data), "original_data": current_data}
            result = agent.run(task, context=context, **kwargs)
            results[agent_name] = result
            # Update for next agent (keep as dict)
            current_data = {"previous_result": result.get("result", ""), "metadata": result.get("metadata", {})}
        
        return {
            "results": results,
            "conductor": "pipeline",
            "agents_used": list(self.agents.keys()),
            "final_result": current_data,
        }
    
    def _determine_agent_order(self, task: str) -> List[str]:
        """
        Determine which agents to use and in what order.
        
        In production, this would use an LLM or routing logic.
        """
        # Simplified: use all agents in order
        return list(self.agents.keys())

