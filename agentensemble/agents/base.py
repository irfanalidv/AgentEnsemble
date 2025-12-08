"""
Base Agent Class

Defines the interface and common functionality for all agents.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """Base state model for agents"""
    
    query: str
    context: Dict[str, Any] = Field(default_factory=dict)
    iteration_count: int = 0
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    result: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseAgent(ABC):
    """
    Base class for all agents in AgentEnsemble.
    
    All agent implementations should inherit from this class.
    """
    
    def __init__(
        self,
        name: str,
        tools: Optional[List[Any]] = None,
        max_iterations: int = 10,
        **kwargs
    ):
        """
        Initialize the agent.
        
        Args:
            name: Name of the agent
            tools: List of tools available to the agent
            max_iterations: Maximum number of iterations
            **kwargs: Additional configuration
        """
        self.name = name
        self.tools = tools or []
        self.max_iterations = max_iterations
        self.config = kwargs
    
    @abstractmethod
    def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Execute the agent with a given query.
        
        Args:
            query: The input query
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing the result and metadata
        """
        pass
    
    def _validate_state(self, state: AgentState) -> bool:
        """Validate agent state"""
        return state.iteration_count < self.max_iterations
    
    def _update_state(
        self,
        state: AgentState,
        result: Optional[str] = None,
        tool_calls: Optional[List[Dict]] = None,
        **metadata
    ) -> AgentState:
        """Update agent state"""
        if result:
            state.result = result
        if tool_calls:
            state.tool_calls.extend(tool_calls)
        state.iteration_count += 1
        state.metadata.update(metadata)
        return state

