"""
Tool Registry

Centralized tool management for agents.
"""

from typing import Any, Dict, List, Optional


class ToolRegistry:
    """
    Registry for managing and organizing tools.
    
    Provides centralized tool discovery and management.
    """
    
    def __init__(self):
        """Initialize tool registry"""
        self._tools: Dict[str, Any] = {}
    
    def register(self, tool: Any, name: Optional[str] = None) -> None:
        """
        Register a tool.
        
        Args:
            tool: Tool instance
            name: Optional name (uses tool.name if available)
        """
        tool_name = name or getattr(tool, "name", f"tool_{len(self._tools)}")
        self._tools[tool_name] = tool
    
    def register_many(self, tools: List[Any]) -> None:
        """
        Register multiple tools.
        
        Args:
            tools: List of tool instances
        """
        for tool in tools:
            self.register(tool)
    
    def get_tool(self, name: str) -> Any:
        """
        Get a tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool instance
        """
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found in registry")
        return self._tools[name]
    
    def get_tools(self) -> List[Any]:
        """
        Get all registered tools.
        
        Returns:
            List of all tools
        """
        return list(self._tools.values())
    
    def list_tools(self) -> List[str]:
        """
        List all tool names.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
    
    def unregister(self, name: str) -> None:
        """
        Unregister a tool.
        
        Args:
            name: Tool name
        """
        if name in self._tools:
            del self._tools[name]

