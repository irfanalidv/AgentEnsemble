"""Tests for tools."""

import pytest

from agentensemble.tools import (
    function_tool,
    FunctionTool,
    get_tool_schemas,
    invoke_tool,
    ToolRegistry,
)
from agentensemble.tools.search import SearchTool


class TestFunctionTool:
    """Tests for @function_tool and FunctionTool."""

    def test_decorator_creates_function_tool(self):
        @function_tool(description="Test")
        def foo(x: str) -> str:
            return x

        assert isinstance(foo, FunctionTool)
        assert foo.name == "foo"

    def test_get_schema(self):
        @function_tool(description="Add two numbers")
        def add(a: int, b: int) -> int:
            return a + b

        schema = add.get_schema()
        assert schema.name == "add"
        assert schema.description == "Add two numbers"
        assert "properties" in schema.parameters
        assert "a" in schema.parameters["properties"]
        assert "b" in schema.parameters["properties"]

    def test_arun_sync_function(self):
        @function_tool()
        def sync_fn(x: str) -> str:
            return x.upper()

        import asyncio
        result = asyncio.run(sync_fn.arun(x="hello"))
        assert result == "HELLO"


class TestToolAdapters:
    """Tests for tool adapters."""

    def test_get_tool_schemas_from_function_tool(self):
        @function_tool(description="Test")
        def t(x: str) -> str:
            return x

        schemas = get_tool_schemas([t])
        assert len(schemas) == 1
        assert schemas[0].name == "t"

    def test_get_tool_schemas_from_legacy_tool(self):
        search = SearchTool()
        schemas = get_tool_schemas([search])
        assert len(schemas) == 1
        assert schemas[0].name == "search"


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = SearchTool()
        registry.register(tool)
        assert registry.get_tool("search") is tool
        assert "search" in registry.list_tools()

    def test_register_many(self):
        registry = ToolRegistry()
        registry.register_many([SearchTool()])
        assert len(registry.get_tools()) == 1
        assert "search" in registry.list_tools()
