"""
Function Tool Decorator

Create tools from plain Python functions with automatic schema generation.
Cleaner than OpenAI's approach: single decorator, no boilerplate.

Usage:
    @function_tool(description="Search the web for information")
    def search(query: str) -> str:
        '''Search the web.'''
        return run_search(query)
"""

import asyncio
from functools import wraps
from typing import Any, Callable, Optional

from agentensemble.llm.interface import ToolSchema
from agentensemble.tools.protocol import _infer_schema_from_func


class FunctionTool:
    """
    Tool wrapping a Python function for LLM function calling.

    Automatically generates JSON Schema from signature and docstring.
    Supports both sync and async functions.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        self._func = func
        self._schema = _infer_schema_from_func(func, name=name, description=description)
        self.name = self._schema.name
        self.description = self._schema.description
        self._is_async = asyncio.iscoroutinefunction(func)

    def get_schema(self) -> ToolSchema:
        return self._schema

    def run(self, **kwargs: Any) -> Any:
        """Execute synchronously."""
        return self._func(**kwargs)

    async def arun(self, **kwargs: Any) -> Any:
        """Execute asynchronously."""
        if self._is_async:
            return await self._func(**kwargs)
        return await asyncio.to_thread(self._func, **kwargs)

    def __call__(self, **kwargs: Any) -> Any:
        return self.run(**kwargs)


def function_tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable[[Callable[..., Any]], FunctionTool]:
    """
    Decorator to create a FunctionTool from a Python function.

    Args:
        name: Tool name (default: inferred from function name)
        description: Tool description (default: first line of docstring)

    Returns:
        FunctionTool instance

    Example:
        @function_tool(description="Search the web")
        def search(query: str) -> str:
            return duckduckgo_search(query)
    """

    def decorator(func: Callable[..., Any]) -> FunctionTool:
        return FunctionTool(func, name=name, description=description)

    return decorator
