"""
RAG-Enhanced Agent

Agent with RAG capabilities and fallback strategies.
"""

from typing import Any, Dict, List, Optional
from agentensemble.agents.base import BaseAgent, AgentState


class RAGAgent(BaseAgent):
    """
    RAG-Enhanced Agent with fallback strategies.
    
    Implements scraping → embedding → retrieval with query variations.
    """
    
    def __init__(
        self,
        name: str = "rag_agent",
        tools: Optional[list] = None,
        max_iterations: int = 10,
        fallback_strategies: int = 3,
        **kwargs
    ):
        """
        Initialize RAG agent.
        
        Args:
            name: Agent name
            tools: Available tools (should include RAG tool)
            max_iterations: Maximum iterations
            fallback_strategies: Number of fallback query variations
            **kwargs: Additional configuration
        """
        super().__init__(name, tools, max_iterations, **kwargs)
        self.fallback_strategies = fallback_strategies
    
    def run(self, query: str, urls: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute RAG agent with fallback strategies.
        
        Args:
            query: Input query
            urls: Optional URLs to use for RAG
            **kwargs: Additional parameters
            
        Returns:
            Result dictionary
        """
        state = AgentState(query=query, context=kwargs.get("context", {}))
        
        # Try primary query
        result = self._query_rag(state.query, urls)
        
        if result and result != "not found":
            state.result = result
        else:
            # Try fallback strategies
            for i in range(self.fallback_strategies):
                fallback_query = self._generate_fallback_query(state.query, i)
                result = self._query_rag(fallback_query, urls)
                if result and result != "not found":
                    state.result = result
                    break
        
        return {
            "result": state.result or "No result found",
            "metadata": {
                "fallback_attempts": min(self.fallback_strategies, state.iteration_count),
                "agent": self.name,
            }
        }
    
    def _query_rag(self, query: str, urls: Optional[List[str]]) -> str:
        """Query RAG system"""
        # Find RAG tool
        rag_tool = next((t for t in self.tools if hasattr(t, "name") and t.name == "rag"), None)
        if rag_tool:
            return rag_tool.run(query, urls=urls)
        return "RAG tool not available"
    
    def _generate_fallback_query(self, original_query: str, attempt: int) -> str:
        """Generate fallback query variation"""
        # Simplified - in production, use LLM to rephrase
        variations = [
            f"More details about: {original_query}",
            f"Alternative phrasing: {original_query}",
            f"Related information: {original_query}",
        ]
        return variations[attempt % len(variations)]

