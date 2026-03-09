"""
Router Agent

LLM-based routing to select the best agent for a given task.
"""

from typing import Any, Dict, List, Optional

from agentensemble.agents.base import BaseAgent
from agentensemble.core.protocol import AgentProtocol
from agentensemble.llm.interface import LLMMessage, LLMProvider, ToolSchema

try:
    from agentensemble.llm.mistral_provider import MistralLLMProvider
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    MistralLLMProvider = None


ROUTE_AGENT_SCHEMA = ToolSchema(
    name="route_to_agent",
    description="Select the best agent for this task based on the query and available agents.",
    parameters={
        "type": "object",
        "properties": {
            "agent_name": {
                "type": "string",
                "description": "Name of the agent to route to",
            },
            "reason": {
                "type": "string",
                "description": "Brief reason for this routing decision",
            },
        },
        "required": ["agent_name"],
    },
)


class RouterAgent(BaseAgent):
    """
    Router agent that uses LLM to select the best downstream agent.

    Use with Ensemble for intelligent supervisor-style routing.
    """

    def __init__(
        self,
        name: str = "router",
        agents: Optional[Dict[str, AgentProtocol]] = None,
        llm: Optional[LLMProvider] = None,
        **kwargs: Any,
    ):
        super().__init__(name, tools=[], max_iterations=1, **kwargs)
        self.agents = agents or {}
        self.llm = llm
        if self.llm is None and LLM_AVAILABLE and MistralLLMProvider:
            try:
                self.llm = MistralLLMProvider(temperature=0.1)
            except (ValueError, ImportError):
                self.llm = None

    def run(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        import asyncio
        return asyncio.run(self.arun(query, **kwargs))

    async def route_only(self, query: str) -> Optional[str]:
        """Return only the selected agent name, without executing. For Ensemble integration."""
        if not self.llm or not self.agents:
            return next(iter(self.agents.keys()), None) if self.agents else None

        agent_names = list(self.agents.keys())
        if len(agent_names) == 1:
            return agent_names[0]

        schema = ToolSchema(
            name="route_to_agent",
            description="Route to the selected agent",
            parameters={
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "enum": agent_names,
                        "description": "Agent to route to",
                    },
                },
                "required": ["agent_name"],
            },
        )
        messages = [
            LLMMessage(role="system", content="Select the best agent for this task. Reply with agent name."),
            LLMMessage(role="user", content=f"Query: {query}\n\nAgents: {', '.join(agent_names)}"),
        ]
        response = await self.llm.agenerate(messages, tools=[schema], tool_choice="auto")
        if response.tool_calls:
            name = (response.tool_calls[0].arguments.get("agent_name") or "").strip()
            if name in self.agents:
                return name
        return agent_names[0]

    async def arun(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        if not self.llm:
            # Fallback: use first agent
            if self.agents:
                agent = next(iter(self.agents.values()))
                return await agent.arun(query, **kwargs)
            return {"result": "No agents available", "metadata": {"routed_to": None}}

        agent_descriptions = "\n".join(
            f"- {name}: available" for name in self.agents.keys()
        )
        prompt = f"""Query: {query}

Available agents:
{agent_descriptions}

Select the best agent for this task. Reply with the agent name only if unsure."""

        messages = [
            LLMMessage(
                role="system",
                content="You are a routing agent. Select the single best agent for each query. Reply with the agent name from the list.",
            ),
            LLMMessage(role="user", content=prompt),
        ]

        agent_names = list(self.agents.keys())
        if len(agent_names) == 1:
            agent_name = agent_names[0]
            agent = self.agents[agent_name]
            result = await agent.arun(query, **kwargs)
            result.setdefault("metadata", {})["routed_to"] = agent_name
            return result

        schema = ToolSchema(
            name="route_to_agent",
            description="Route to the selected agent",
            parameters={
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "enum": agent_names,
                        "description": "Agent to route to",
                    },
                },
                "required": ["agent_name"],
            },
        )

        response = await self.llm.agenerate(
            messages,
            tools=[schema],
            tool_choice="auto",
        )

        agent_name = None
        if response.tool_calls:
            agent_name = (response.tool_calls[0].arguments.get("agent_name") or "").strip()
        if not agent_name or agent_name not in self.agents:
            agent_name = next(iter(self.agents.keys()), None)

        if agent_name and agent_name in self.agents:
            agent = self.agents[agent_name]
            result = await agent.arun(query, **kwargs)
            result["metadata"] = result.get("metadata", {})
            result["metadata"]["routed_to"] = agent_name
            return result

        return {
            "result": "No suitable agent found",
            "metadata": {"routed_to": None},
        }
