"""
ReAct Agent Implementation

Simple reasoning + acting pattern with tool calling.
"""

from typing import Any, Dict, List, Optional
from agentensemble.agents.base import BaseAgent, AgentState


class ReActAgent(BaseAgent):
    """
    ReAct (Reasoning + Acting) Agent
    
    A simple agent that reasons about actions and uses tools to accomplish tasks.
    """
    
    def __init__(
        self,
        name: str = "react_agent",
        tools: Optional[List[Any]] = None,
        max_iterations: int = 10,
        **kwargs
    ):
        """
        Initialize ReAct agent.
        
        Args:
            name: Agent name
            tools: Available tools
            max_iterations: Maximum reasoning/acting cycles
            **kwargs: Additional configuration
        """
        super().__init__(name, tools, max_iterations, **kwargs)
    
    def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Execute ReAct agent.
        
        Args:
            query: Input query
            **kwargs: Additional parameters
            
        Returns:
            Result dictionary with answer and metadata
        """
        state = AgentState(query=query, context=kwargs.get("context", {}))
        
        # Simple ReAct loop: Think -> Act -> Observe
        for iteration in range(self.max_iterations):
            if not self._validate_state(state):
                break
            
            # Think: Reason about what to do
            action = self._think(state)
            
            if action == "ANSWER":
                break
            
            # Act: Use a tool
            if action == "TOOL":
                tool_result = self._act(state)
                state.context["last_tool_result"] = tool_result
                
                # If we got a good result, use it as the answer
                if tool_result and tool_result != "No tools available" and tool_result != "Tool execution failed":
                    state.result = tool_result
                
                state = self._update_state(
                    state,
                    tool_calls=[{"iteration": iteration, "result": tool_result}]
                )
            else:
                break
        
        # If no result yet, try to synthesize from context
        if not state.result and state.context.get("last_tool_result"):
            state.result = state.context["last_tool_result"]
        
        return {
            "result": state.result or state.context.get("last_tool_result", "No result generated"),
            "metadata": {
                "iterations": state.iteration_count,
                "tool_calls": len(state.tool_calls),
                "agent": self.name,
            }
        }
    
    def _think(self, state: AgentState) -> str:
        """
        Reasoning step: Decide what action to take.
        
        Returns:
            "TOOL" to use a tool, "ANSWER" to provide final answer
        """
        # Simplified logic - in production, this would use an LLM
        if state.result:
            return "ANSWER"
        if state.iteration_count < 3 and self.tools:
            return "TOOL"
        return "ANSWER"
    
    def _act(self, state: AgentState) -> str:
        """
        Acting step: Execute a tool.
        
        Returns:
            Tool result
        """
        if not self.tools:
            return "No tools available"
        
        # Use first available tool (simplified)
        tool = self.tools[0]
        if hasattr(tool, "run"):
            return tool.run(state.query)
        return "Tool execution failed"

