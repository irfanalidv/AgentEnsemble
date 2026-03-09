"""
Ensemble Orchestration Pattern

Coordinates multiple agents working together in harmony.
Supports sync perform() and async aperform() for parallel execution.
"""

import asyncio
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
        router: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize ensemble.

        Args:
            agents: Dictionary of agent name -> agent instance
            conductor: Coordination mode ("supervisor", "swarm", "pipeline")
            router: Optional RouterAgent for LLM-based agent selection (supervisor mode only)
            **kwargs: Additional configuration
        """
        self.agents = agents
        self.conductor = conductor
        self.router = router
        self.config = kwargs
    
    def perform(self, task: str, data: Any = None, **kwargs) -> Dict[str, Any]:
        """
        Execute ensemble of agents on a task (synchronous).

        Args:
            task: Task description
            data: Input data
            **kwargs: Additional parameters

        Returns:
            Combined results from all agents
        """
        return asyncio.run(self.aperform(task, data, **kwargs))

    async def aperform(self, task: str, data: Any = None, **kwargs) -> Dict[str, Any]:
        """
        Execute ensemble of agents on a task (asynchronous).

        Swarm mode runs agents in parallel via asyncio.gather.
        """
        if self.conductor == "supervisor":
            return await self._supervisor_coordinate_async(task, data, **kwargs)
        elif self.conductor == "swarm":
            return await self._swarm_coordinate_async(task, data, **kwargs)
        elif self.conductor == "pipeline":
            return await self._pipeline_coordinate_async(task, data, **kwargs)
        else:
            raise ValueError(f"Unknown conductor mode: {self.conductor}")
    
    def _supervisor_coordinate(
        self,
        task: str,
        data: Any,
        **kwargs
    ) -> Dict[str, Any]:
        """Supervisor pattern: Central agent coordinates others (sync)."""
        return asyncio.run(self._supervisor_coordinate_async(task, data, **kwargs))

    async def _supervisor_coordinate_async(
        self,
        task: str,
        data: Any,
        **kwargs
    ) -> Dict[str, Any]:
        """Supervisor pattern: Central agent coordinates others."""
        results = {}
        agent_order = await self._determine_agent_order_async(task)

        for agent_name in agent_order:
            if agent_name in self.agents:
                agent = self.agents[agent_name]
                context = data if isinstance(data, dict) else {"query": str(data), "original_data": data}
                result = await agent.arun(task, context=context, **kwargs)
                results[agent_name] = result
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
        """Swarm pattern (sync wrapper)."""
        return asyncio.run(self._swarm_coordinate_async(task, data, **kwargs))

    async def _swarm_coordinate_async(
        self,
        task: str,
        data: Any,
        **kwargs
    ) -> Dict[str, Any]:
        """Swarm pattern: Agents run in parallel via asyncio.gather."""
        context = data if isinstance(data, dict) else {"query": str(data), "original_data": data}

        async def run_agent(name: str, agent: BaseAgent) -> tuple[str, Dict[str, Any]]:
            result = await agent.arun(task, context=context, **kwargs)
            return (name, result)

        tasks = [run_agent(name, agent) for name, agent in self.agents.items()]
        pairs = await asyncio.gather(*tasks, return_exceptions=False)
        results = dict(pairs)

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
        """Pipeline pattern (sync wrapper)."""
        return asyncio.run(self._pipeline_coordinate_async(task, data, **kwargs))

    async def _pipeline_coordinate_async(
        self,
        task: str,
        data: Any,
        **kwargs
    ) -> Dict[str, Any]:
        """Pipeline pattern: Sequential agent execution."""
        results = {}
        current_data = data

        for agent_name, agent in self.agents.items():
            context = current_data if isinstance(current_data, dict) else {"query": str(current_data), "original_data": current_data}
            result = await agent.arun(task, context=context, **kwargs)
            results[agent_name] = result
            current_data = {"previous_result": result.get("result", ""), "metadata": result.get("metadata", {})}

        return {
            "results": results,
            "conductor": "pipeline",
            "agents_used": list(self.agents.keys()),
            "final_result": current_data,
        }
    
    async def _determine_agent_order_async(self, task: str) -> List[str]:
        """Determine agent order. Uses RouterAgent.route_only when provided."""
        if self.router and hasattr(self.router, "route_only"):
            name = await self.router.route_only(task)
            if name and name in self.agents:
                return [name]
        return list(self.agents.keys())

