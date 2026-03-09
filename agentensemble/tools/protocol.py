"""
Tool Protocol & Schema

Defines the contract for tools compatible with LLM function calling.
Enables automatic schema generation and consistent invocation across agents.

Research basis:
- OpenAI/Anthropic function calling: JSON Schema for parameters
- LangChain bind_tools: Accepts functions, Pydantic models, BaseTool
"""

from typing import Any, Callable, Dict, Optional, Protocol, runtime_checkable

from pydantic import BaseModel

from agentensemble.llm.interface import ToolSchema


# -----------------------------------------------------------------------------
# Tool Protocol
# -----------------------------------------------------------------------------


@runtime_checkable
class Tool(Protocol):
    """
    Protocol for tools compatible with LLM function calling.

    Tools implement name, description, get_schema(), and run()/arun().
    Enables both sync and async execution.
    """

    name: str
    description: str

    def get_schema(self) -> ToolSchema:
        """Return JSON Schema for function calling."""
        ...

    def run(self, **kwargs: Any) -> Any:
        """Synchronous execution."""
        ...

    async def arun(self, **kwargs: Any) -> Any:
        """Async execution. Default falls back to run in executor."""
        ...


# -----------------------------------------------------------------------------
# Schema Generation
# -----------------------------------------------------------------------------


def _infer_schema_from_func(
    func: Callable[..., Any],
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> ToolSchema:
    """
    Infer JSON Schema from function signature and docstring.

    Uses inspect and typing to build parameters schema.
    Compatible with OpenAI/Anthropic function calling format.
    """
    import inspect
    import json

    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    # First line of docstring as description
    desc = description or doc.split("\n")[0].strip() or f"Execute {func.__name__}"
    tool_name = name or func.__name__

    properties: Dict[str, Any] = {}
    required: list = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

        # Infer type from annotation
        ann = param.annotation
        if ann is inspect.Parameter.empty:
            schema_type = "string"
        elif ann == str:
            schema_type = "string"
        elif ann == int:
            schema_type = "integer"
        elif ann == float:
            schema_type = "number"
        elif ann == bool:
            schema_type = "boolean"
        elif ann == list:
            schema_type = "array"
        elif ann == dict:
            schema_type = "object"
        else:
            schema_type = "string"

        properties[param_name] = {
            "type": schema_type,
            "description": f"Parameter {param_name}",
        }

    return ToolSchema(
        name=tool_name,
        description=desc,
        parameters={
            "type": "object",
            "properties": properties,
            "required": required,
        },
    )
