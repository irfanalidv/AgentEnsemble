"""
Search Tool

Web search functionality for agents using LangChain community tools.
Defaults to Serper API if API key is available, otherwise falls back to DuckDuckGo.
"""

import os
from typing import Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from langchain_community.tools import (
        DuckDuckGoSearchRun,
        DuckDuckGoSearchResults,
    )

    # Optional: Serper API (requires API key) - only import if needed
    try:
        from langchain_community.utilities import GoogleSerperAPIWrapper
        from langchain_core.tools import Tool

        SERPER_AVAILABLE = True
    except ImportError:
        SERPER_AVAILABLE = False
        GoogleSerperAPIWrapper = None
        Tool = None
    from langchain_core.tools import BaseTool

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseTool = object
    SERPER_AVAILABLE = False
    GoogleSerperAPIWrapper = None
    Tool = None


class SearchTool:
    """
    Web search tool for agents using LangChain community tools.

    **Default: Serper API** (if SERPER_API_KEY is in .env)
    **Fallback: DuckDuckGo** (free, open-source, no API key required)

    Supports multiple providers:
    - Serper API (default if API key available, requires SERPER_API_KEY in .env)
    - DuckDuckGo (fallback, free, open-source, no API key required) ⭐
    - DuckDuckGo Results (structured results, free)
    """

    def __init__(self, provider: Optional[str] = None, api_key: Optional[str] = None, **kwargs):
        """
        Initialize search tool.

        Args:
            provider: Search provider
                - None (auto-detect: Serper if API key available, else DuckDuckGo)
                - "serper" (requires SERPER_API_KEY in .env or api_key parameter)
                - "duckduckgo" (free, open-source) ⭐
                - "duckduckgo_results" (structured results, free)
            api_key: API key for Serper (defaults to SERPER_API_KEY from .env)
            **kwargs: Additional configuration
        """
        self.name = "search"

        # Auto-detect provider if not specified
        if provider is None:
            serper_key = api_key or os.getenv("SERPER_API_KEY")
            if serper_key and SERPER_AVAILABLE:
                provider = "serper"
            else:
                provider = "duckduckgo"

        self.provider = provider
        # Use provided api_key or get from environment
        self.api_key = api_key or os.getenv("SERPER_API_KEY")
        self._tool: Optional[BaseTool] = None
        self._initialize_tool()

    def _initialize_tool(self) -> None:
        """Initialize the underlying LangChain tool"""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-community is required for SearchTool. "
                "Install with: pip install langchain-community"
            )

        if self.provider == "serper":
            if not SERPER_AVAILABLE or not GoogleSerperAPIWrapper:
                # Fallback to DuckDuckGo if Serper not available
                print("Warning: Serper API not available, falling back to DuckDuckGo")
                self.provider = "duckduckgo"
                self._tool = DuckDuckGoSearchRun()
                return

            if not self.api_key:
                # Fallback to DuckDuckGo if no API key
                print("Warning: SERPER_API_KEY not found, falling back to DuckDuckGo")
                self.provider = "duckduckgo"
                self._tool = DuckDuckGoSearchRun()
                return

            # Create Serper API tool
            serper = GoogleSerperAPIWrapper(serper_api_key=self.api_key)
            self._tool = Tool(
                name="serper_search", description="Search the web using Serper API", func=serper.run
            )
        elif self.provider == "duckduckgo":
            self._tool = DuckDuckGoSearchRun()
        elif self.provider == "duckduckgo_results":
            self._tool = DuckDuckGoSearchResults()
        else:
            raise ValueError(
                f"Unknown provider: {self.provider}. "
                "Supported: 'serper' (default if API key available), 'duckduckgo' (free, fallback), 'duckduckgo_results' (free)"
            )

    def run(self, query: str, **kwargs) -> str:
        """
        Execute search.

        Args:
            query: Search query
            **kwargs: Additional parameters

        Returns:
            Search results
        """
        if not self._tool:
            self._initialize_tool()

        if self._tool:
            # Use LangChain tool's invoke method
            result = self._tool.invoke(query)
            return str(result)
        return f"Search results for: {query}"

    def __call__(self, query: str, **kwargs) -> str:
        """Make tool callable"""
        return self.run(query, **kwargs)

    def as_langchain_tool(self) -> BaseTool:
        """
        Return the underlying LangChain tool for direct use.

        Returns:
            LangChain BaseTool instance
        """
        if not self._tool:
            self._initialize_tool()
        return self._tool
