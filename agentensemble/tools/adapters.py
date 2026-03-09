"""
Tool Adapters

Adapt existing tools (SearchTool, RAGTool, etc.) to the Tool protocol
for use with LLM function calling.
"""

from typing import Any, Dict, Optional

from agentensemble.llm.interface import ToolSchema


def adapt_to_tool_schema(
    tool: Any,
    name: Optional[str] = None,
    description: Optional[str] = None,
    param_schema: Optional[Dict[str, Any]] = None,
) -> ToolSchema:
    """
    Create ToolSchema from a legacy tool (SearchTool, RAGTool, etc.).

    Args:
        tool: Tool instance with .name and .run()
        name: Override name (default: tool.name)
        description: Override description
        param_schema: Custom parameter schema

    Returns:
        ToolSchema for LLM function calling
    """
    tool_name = name or getattr(tool, "name", "unknown")
    tool_desc = description or getattr(tool, "description", f"Tool: {tool_name}")

    # Default schema for common tools
    if param_schema is None:
        if tool_name in ("search", "serper_search"):
            param_schema = {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            }
        elif tool_name == "rag":
            param_schema = {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "RAG query"},
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional URLs to scrape",
                    },
                },
                "required": ["question"],
            }
        else:
            param_schema = {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Input query"},
                },
                "required": ["query"],
            }

    return ToolSchema(
        name=tool_name,
        description=tool_desc,
        parameters=param_schema,
    )


def get_tool_schemas(tools: list) -> list:
    """
    Convert a list of tools to ToolSchema list.

    Handles both:
    - FunctionTool / protocol-compliant tools
    - Legacy tools (SearchTool, RAGTool) via adapt_to_tool_schema
    """
    schemas = []
    for t in tools:
        if hasattr(t, "get_schema"):
            schemas.append(t.get_schema())
        else:
            schemas.append(adapt_to_tool_schema(t))
    return schemas


def _normalize_arguments(tool: Any, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize arguments for legacy tools (e.g., query->question for RAGTool)."""
    args = dict(arguments)
    if tool_name == "rag" and "query" in args and "question" not in args:
        args["question"] = args.pop("query")
    return args


def invoke_tool(tool: Any, name: str, arguments: Dict[str, Any]) -> Any:
    """
    Invoke a tool by name with given arguments.

    Works with legacy tools (run) and FunctionTool (run/arun).
    """
    args = _normalize_arguments(tool, name, arguments)
    if hasattr(tool, "run"):
        return tool.run(**args)
    return str(arguments)


async def ainvoke_tool(tool: Any, name: str, arguments: Dict[str, Any]) -> Any:
    """Async invoke tool."""
    args = _normalize_arguments(tool, name, arguments)
    if hasattr(tool, "arun"):
        return await tool.arun(**args)
    if hasattr(tool, "run"):
        import asyncio
        return await asyncio.to_thread(tool.run, **args)
    return str(arguments)
