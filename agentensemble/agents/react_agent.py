"""
ReAct Agent Implementation

LLM-powered reasoning + acting pattern with native tool calling.
Uses function calling (not text parsing) per 2024 best practices.

Research basis:
- ReAct: Synergizing Reasoning and Acting (Yao et al., 2022)
- Native tool/function calling preferred over THOUGHT/ACTION parsing (LangChain 2024)
"""

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional

from agentensemble.agents.base import BaseAgent, AgentState
from agentensemble.llm.interface import LLMMessage, LLMProvider, ToolCall
from agentensemble.tools.adapters import get_tool_schemas, ainvoke_tool

try:
    from agentensemble.llm.mistral_provider import MistralLLMProvider
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    MistralLLMProvider = None


class ReActAgent(BaseAgent):
    """
    ReAct (Reasoning + Acting) Agent with LLM-powered tool selection.

    Uses native function calling for tool selection—the LLM decides
    which tool to call and with what parameters. No hardcoded rules.
    """

    def __init__(
        self,
        name: str = "react_agent",
        tools: Optional[List[Any]] = None,
        max_iterations: int = 10,
        llm: Optional[LLMProvider] = None,
        session: Optional[Any] = None,
        **kwargs: Any,
    ):
        """
        Initialize ReAct agent.

        Args:
            name: Agent name
            tools: Available tools (SearchTool, RAGTool, FunctionTool, etc.)
            max_iterations: Maximum reasoning/acting cycles
            llm: LLM provider (default: MistralLLMProvider if available)
            session: Optional Session for multi-turn conversation history
            **kwargs: Additional configuration
        """
        super().__init__(name, tools, max_iterations, **kwargs)
        self.llm = llm
        self.session = session
        if self.llm is None and LLM_AVAILABLE and MistralLLMProvider:
            self.llm = MistralLLMProvider(temperature=0.2)

    def run(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute ReAct agent (synchronous).

        Args:
            query: Input query
            **kwargs: Additional parameters (context, etc.)

        Returns:
            Result dictionary with answer and metadata
        """
        return asyncio.run(self._run_loop(query, **kwargs))

    async def arun(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute ReAct agent (asynchronous).

        Args:
            query: Input query
            **kwargs: Additional parameters

        Returns:
            Result dictionary with answer and metadata
        """
        return await self._run_loop(query, **kwargs)

    async def astream(
        self, query: str, **kwargs: Any
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream execution events for real-time UX.

        Yields: {"type": "thinking"}, {"type": "tool_start", "name": "..."},
                {"type": "tool_end", "name": "...", "result": "..."},
                {"type": "done", "result": "...", "metadata": {...}}
        """
        async for event in self._stream_loop(query, **kwargs):
            yield event

    async def _stream_loop(
        self, query: str, **kwargs: Any
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream events during ReAct loop."""
        if self.llm is None:
            raise RuntimeError(
                "LLM required for ReActAgent. Install langchain-mistralai and set MISTRAL_API_KEY."
            )

        state = AgentState(query=query, context=kwargs.get("context", {}))
        tool_schemas = get_tool_schemas(self.tools) if self.tools else []
        tool_map = {t.name: t for t in self.tools} if self.tools else {}

        system_prompt = self.llm.get_react_system_prompt(tool_schemas, self.max_iterations)
        messages: List[LLMMessage] = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=query),
        ]

        for iteration in range(self.max_iterations):
            if not self._validate_state(state):
                break

            has_tool_results = any(m.role == "tool" for m in messages)
            use_tools = tool_schemas and not has_tool_results

            yield {"type": "thinking", "iteration": iteration}
            response = await self.llm.agenerate(
                messages,
                tools=tool_schemas if use_tools else None,
                tool_choice="auto" if use_tools else None,
            )

            if response.tool_calls:
                messages.append(
                    LLMMessage(
                        role="assistant",
                        content=response.content or "",
                        tool_calls=response.tool_calls,
                    )
                )
                for tc in response.tool_calls:
                    yield {"type": "tool_start", "name": tc.name, "arguments": tc.arguments}
                    tool = tool_map.get(tc.name)
                    if tool:
                        try:
                            result = await ainvoke_tool(tool, tc.name, tc.arguments)
                            result_str = str(result)[:2000]
                        except Exception as e:
                            result_str = f"Error: {e}"
                    else:
                        result_str = f"Tool '{tc.name}' not found"

                    messages.append(
                        LLMMessage(
                            role="tool",
                            content=result_str,
                            tool_call_id=tc.id or f"call_{tc.name}",
                        )
                    )
                    state = self._update_state(
                        state,
                        tool_calls=[{"name": tc.name, "result": result_str}],
                    )
                    yield {"type": "tool_end", "name": tc.name, "result": result_str}
            else:
                state.result = response.content or state.result
                break

        yield {
            "type": "done",
            "result": state.result or "No result generated",
            "metadata": {
                "iterations": state.iteration_count,
                "tool_calls": len(state.tool_calls),
                "agent": self.name,
            },
        }

    def _session_to_messages(self, limit: int = 20) -> List[LLMMessage]:
        """Load session history and convert to LLMMessage list."""
        if not self.session:
            return []
        raw = self.session.get_messages(limit=limit)
        out: List[LLMMessage] = []
        for m in raw:
            tc = None
            if m.get("tool_calls"):
                tc = [
                    ToolCall(id=x.get("id", ""), name=x.get("name", ""), arguments=x.get("arguments", {}))
                    for x in m["tool_calls"]
                ]
            out.append(
                LLMMessage(
                    role=m["role"],
                    content=m.get("content", ""),
                    tool_call_id=m.get("tool_call_id"),
                    tool_calls=tc,
                )
            )
        return out

    def _messages_to_session(self, to_save: List[LLMMessage]) -> None:
        """Save messages to session."""
        if not self.session:
            return
        raw = []
        for m in to_save:
            d: Dict[str, Any] = {"role": m.role, "content": m.content if isinstance(m.content, str) else ""}
            if m.tool_call_id:
                d["tool_call_id"] = m.tool_call_id
            if m.tool_calls:
                d["tool_calls"] = [{"id": t.id, "name": t.name, "arguments": t.arguments} for t in m.tool_calls]
            raw.append(d)
        self.session.add_messages(raw)

    async def _run_loop(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        """Async ReAct loop: Reason -> Act -> Observe until final answer."""
        if self.llm is None:
            raise RuntimeError(
                "LLM required for ReActAgent. Install langchain-mistralai and set MISTRAL_API_KEY, "
                "or pass llm=YourLLMProvider()."
            )

        state = AgentState(query=query, context=kwargs.get("context", {}))
        tool_schemas = get_tool_schemas(self.tools) if self.tools else []
        tool_map = {t.name: t for t in self.tools} if self.tools else {}

        system_prompt = self.llm.get_react_system_prompt(tool_schemas, self.max_iterations)
        history = self._session_to_messages()
        messages: List[LLMMessage] = [
            LLMMessage(role="system", content=system_prompt),
            *history,
            LLMMessage(role="user", content=query),
        ]

        for iteration in range(self.max_iterations):
            if not self._validate_state(state):
                break

            # After tool results, omit tools to encourage final answer
            has_tool_results = any(m.role == "tool" for m in messages)
            use_tools = tool_schemas and not has_tool_results

            response = await self.llm.agenerate(
                messages,
                tools=tool_schemas if use_tools else None,
                tool_choice="auto" if use_tools else None,
            )

            if response.tool_calls:
                # Append assistant message with tool_calls (required for multi-turn)
                messages.append(
                    LLMMessage(
                        role="assistant",
                        content=response.content or "",
                        tool_calls=response.tool_calls,
                    )
                )
                # Execute each tool and append results (preserve tool_call id for matching)
                for tc in response.tool_calls:
                    tool = tool_map.get(tc.name)
                    if tool:
                        try:
                            result = await ainvoke_tool(tool, tc.name, tc.arguments)
                            result_str = str(result)[:2000]
                        except Exception as e:
                            result_str = f"Error: {e}"
                    else:
                        result_str = f"Tool '{tc.name}' not found"

                    messages.append(
                        LLMMessage(
                            role="tool",
                            content=result_str,
                            tool_call_id=tc.id or f"call_{tc.name}",
                        )
                    )
                    state = self._update_state(
                        state,
                        tool_calls=[{"name": tc.name, "result": result_str}],
                    )
            else:
                # Final answer
                state.result = response.content or state.result
                break

        # Persist to session (user + assistant/tool messages from this turn)
        if self.session:
            to_save = messages[len(history) + 2:]  # Skip system, history; keep user + new
            if to_save:
                self._messages_to_session(to_save)

        return {
            "result": state.result or "No result generated",
            "metadata": {
                "iterations": state.iteration_count,
                "tool_calls": len(state.tool_calls),
                "agent": self.name,
            },
        }
