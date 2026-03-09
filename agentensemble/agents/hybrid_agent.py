"""
Hybrid Agent Implementation

Advanced agent with iterative refinement and early stopping.
Supports LLM-based routing (Mistral) or fixed pipeline fallback.
"""

import asyncio
from typing import Any, Dict, List, Optional

from agentensemble.agents.base import BaseAgent, AgentState
from agentensemble.llm.interface import LLMMessage, LLMProvider, ToolSchema

try:
    from agentensemble.llm.mistral_provider import MistralLLMProvider
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    MistralLLMProvider = None


# Routing tool: LLM chooses next action via function calling
ROUTE_ACTION_SCHEMA = ToolSchema(
    name="choose_next_action",
    description="Choose the next step in the hybrid workflow. SEARCH: search the web for current info. RAG: retrieve from documents/context. VALIDATE: validate the current answer quality. ANSWER: return the final answer to the user.",
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["SEARCH", "RAG", "VALIDATE", "ANSWER"],
                "description": "The next action to take",
            },
            "reason": {
                "type": "string",
                "description": "Brief reason for this choice",
            },
        },
        "required": ["action"],
    },
)


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
        llm: Optional[LLMProvider] = None,
        **kwargs
    ):
        """
        Initialize Hybrid agent.

        Args:
            name: Agent name
            tools: Available tools (search, rag, validator)
            max_iterations: Maximum iterations
            early_stopping: Enable early stopping when answer found
            llm: LLM provider for routing (default: MistralLLMProvider if available).
                 If None, uses fixed pipeline: SEARCH -> RAG -> VALIDATE -> ANSWER
            **kwargs: Additional configuration
        """
        super().__init__(name, tools, max_iterations, **kwargs)
        self.early_stopping = early_stopping
        self.llm = llm
        if self.llm is None and LLM_AVAILABLE and MistralLLMProvider:
            try:
                self.llm = MistralLLMProvider(temperature=0.2)
            except (ValueError, ImportError):
                self.llm = None  # Fall back to fixed pipeline if no API key
    
    def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Execute Hybrid agent with intelligent routing.

        Args:
            query: Input query
            **kwargs: Additional parameters

        Returns:
            Result dictionary
        """
        return asyncio.run(self._run_loop(query, **kwargs))

    async def arun(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Execute Hybrid agent asynchronously.

        Args:
            query: Input query
            **kwargs: Additional parameters

        Returns:
            Result dictionary
        """
        return await self._run_loop(query, **kwargs)

    async def _run_loop(self, query: str, **kwargs) -> Dict[str, Any]:
        """Main hybrid workflow loop."""
        state = AgentState(query=query, context=kwargs.get("context", {}))
        actions_taken: List[str] = []

        for iteration in range(self.max_iterations):
            if not self._validate_state(state):
                break

            # Determine next action (LLM or fixed pipeline)
            action = await self._decide_action(state)

            if action == "ANSWER":
                actions_taken.append(action)
                if self.early_stopping:
                    break
            elif action == "SEARCH":
                result = self._search(state)
                state.context["search_results"] = result
                actions_taken.append(action)
            elif action == "RAG":
                result = self._rag(state)
                state.context["rag_results"] = result
                actions_taken.append(action)
            elif action == "VALIDATE":
                validation = self._validate(state)
                state.context["validation"] = validation
                actions_taken.append(action)
            else:
                break

            state = self._update_state(state, iteration=iteration)

        # Final answer synthesis
        if not state.result:
            state.result = self._synthesize_answer(state)

        state.context["actions"] = actions_taken

        return {
            "result": state.result or "No result generated",
            "metadata": {
                "iterations": state.iteration_count,
                "actions_taken": actions_taken,
                "agent": self.name,
            },
        }
    
    async def _decide_action(self, state: AgentState) -> str:
        """
        Decide next action: LLM routing (if llm set) or fixed pipeline.

        Returns:
            Next action: "SEARCH", "RAG", "VALIDATE", "ANSWER"
        """
        if self.llm:
            return await self._decide_action_llm(state)
        return self._decide_action_fixed(state)

    def _decide_action_fixed(self, state: AgentState) -> str:
        """Fixed pipeline: SEARCH -> RAG -> VALIDATE -> ANSWER."""
        if state.result:
            return "ANSWER"
        if state.iteration_count == 0:
            return "SEARCH"
        if state.iteration_count == 1:
            return "RAG"
        if state.iteration_count == 2:
            return "VALIDATE"
        return "ANSWER"

    def _build_routing_prompt(self, state: AgentState) -> str:
        """Build context for LLM routing decision."""
        parts = [
            f"Query: {state.query}",
            f"Iteration: {state.iteration_count}",
        ]
        if state.context.get("search_results"):
            sr = state.context["search_results"]
            preview = str(sr)[:500] + "..." if len(str(sr)) > 500 else str(sr)
            parts.append(f"Search results (available): {preview}")
        else:
            parts.append("Search results: not yet run")
        if state.context.get("rag_results"):
            rr = state.context["rag_results"]
            preview = str(rr)[:500] + "..." if len(str(rr)) > 500 else str(rr)
            parts.append(f"RAG results (available): {preview}")
        else:
            parts.append("RAG results: not yet run")
        if state.context.get("validation"):
            parts.append("Validation: already run")
        else:
            parts.append("Validation: not yet run")
        return "\n".join(parts)

    async def _decide_action_llm(self, state: AgentState) -> str:
        """Use LLM to choose next action via function calling."""
        prompt = self._build_routing_prompt(state)
        messages: List[LLMMessage] = [
            LLMMessage(
                role="system",
                content="You are a routing agent. Choose the next step: SEARCH (get web info), RAG (retrieve from docs), VALIDATE (check answer quality), or ANSWER (return final answer). Pick the most useful next step given what has already been done.",
            ),
            LLMMessage(role="user", content=prompt),
        ]

        response = await self.llm.agenerate(
            messages,
            tools=[ROUTE_ACTION_SCHEMA],
            tool_choice="required",
        )

        if response.tool_calls:
            tc = response.tool_calls[0]
            action = (tc.arguments.get("action") or "ANSWER").upper()
            if action in ("SEARCH", "RAG", "VALIDATE", "ANSWER"):
                return action
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

