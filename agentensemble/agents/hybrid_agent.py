"""
Hybrid Agent Implementation

Advanced agent with iterative refinement and early stopping.
"""

from typing import Any, Dict, Optional
from agentensemble.agents.base import BaseAgent, AgentState


class HybridAgent(BaseAgent):
    """
    Hybrid Agent with intelligent routing and iterative refinement.
    
    Combines multiple strategies: analysis → search → RAG → validation
    """
    
    def __init__(
        self,
        name: str = "hybrid_agent",
        tools: Optional[list] = None,
        max_iterations: int = 15,
        early_stopping: bool = True,
        **kwargs
    ):
        """
        Initialize Hybrid agent.
        
        Args:
            name: Agent name
            tools: Available tools
            max_iterations: Maximum iterations
            early_stopping: Enable early stopping when answer found
            **kwargs: Additional configuration
        """
        super().__init__(name, tools, max_iterations, **kwargs)
        self.early_stopping = early_stopping
    
    def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Execute Hybrid agent with intelligent routing.
        
        Args:
            query: Input query
            **kwargs: Additional parameters
            
        Returns:
            Result dictionary
        """
        state = AgentState(query=query, context=kwargs.get("context", {}))
        
        # Hybrid workflow: Analyze → Search → RAG → Validate
        for iteration in range(self.max_iterations):
            if not self._validate_state(state):
                break
            
            # Determine next action
            action = self._decide_action(state)
            
            if action == "ANSWER":
                if self.early_stopping:
                    break
            elif action == "SEARCH":
                result = self._search(state)
                state.context["search_results"] = result
            elif action == "RAG":
                result = self._rag(state)
                state.context["rag_results"] = result
            elif action == "VALIDATE":
                validation = self._validate(state)
                state.context["validation"] = validation
            else:
                break
            
            state = self._update_state(state, iteration=iteration)
        
        # Final answer synthesis
        if not state.result:
            state.result = self._synthesize_answer(state)
        
        return {
            "result": state.result or "No result generated",
            "metadata": {
                "iterations": state.iteration_count,
                "actions_taken": state.context.get("actions", []),
                "agent": self.name,
            }
        }
    
    def _decide_action(self, state: AgentState) -> str:
        """
        Intelligent action decision.
        
        Returns:
            Next action: "SEARCH", "RAG", "VALIDATE", "ANSWER"
        """
        # Simplified logic - in production, use LLM for routing
        if state.result:
            return "ANSWER"
        if state.iteration_count == 0:
            return "SEARCH"
        if state.iteration_count == 1:
            return "RAG"
        if state.iteration_count == 2:
            return "VALIDATE"
        return "ANSWER"
    
    def _search(self, state: AgentState) -> str:
        """Execute search"""
        search_tool = next((t for t in self.tools if hasattr(t, "name") and t.name == "search"), None)
        if search_tool:
            return search_tool.run(state.query)
        return "Search tool not available"
    
    def _rag(self, state: AgentState) -> str:
        """Execute RAG"""
        rag_tool = next((t for t in self.tools if hasattr(t, "name") and t.name == "rag"), None)
        if rag_tool:
            # Handle search_results as either string or dict
            search_results = state.context.get("search_results", "")
            if isinstance(search_results, dict):
                urls = search_results.get("urls", [])
            else:
                urls = []  # No URLs from search
            return rag_tool.run(state.query, urls=urls)
        return "RAG tool not available"
    
    def _validate(self, state: AgentState) -> Dict[str, Any]:
        """Execute validation"""
        validator = next((t for t in self.tools if hasattr(t, "name") and t.name == "validator"), None)
        if validator:
            result = state.context.get("rag_results", "")
            return validator.run(result, context=state.context)
        return {"valid": True, "confidence": 0.8}
    
    def _synthesize_answer(self, state: AgentState) -> str:
        """Synthesize final answer from all results"""
        # Combine search and RAG results
        search_result = state.context.get("search_results", "")
        rag_result = state.context.get("rag_results", "")
        
        if rag_result and rag_result != "RAG tool not available":
            return rag_result
        if search_result and search_result != "Search tool not available":
            return search_result
        return "Unable to generate answer"

