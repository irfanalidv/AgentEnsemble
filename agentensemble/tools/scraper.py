"""
Scraper Tool

Web page fetching for agents.

- ``provider="simple"`` (default): HTTP GET with ``httpx`` and a timeout, HTML stripped to text
  with the standard-library parser. No browser required.
- ``provider="playwright"``: LangChain's Playwright ``NavigateTool`` for JavaScript-heavy pages.
"""

from __future__ import annotations

import logging
import re
from html.parser import HTMLParser
from typing import Any

import httpx

try:
    from langchain_community.tools.playwright import NavigateTool
    from langchain_core.tools import BaseTool

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    NavigateTool = None  # type: ignore[assignment,misc]
    BaseTool = object  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

_PROVIDERS = ("simple", "playwright")
_SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "head"}
_BLOCK_TAGS = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section"}

DEFAULT_TIMEOUT = 15.0
DEFAULT_MAX_CHARS = 20_000
DEFAULT_USER_AGENT = "agentensemble-scraper (+https://github.com/irfanalidv/AgentEnsemble)"


class ScraperError(RuntimeError):
    """Raised by :meth:`ScraperTool.fetch` when a page cannot be retrieved."""


class _TextExtractor(HTMLParser):
    """Collect visible text from an HTML document."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts)
        lines = (re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in raw.splitlines())
        return "\n".join(line for line in lines if line)


def html_to_text(html: str) -> str:
    """Convert an HTML string to readable plain text (scripts and styles removed)."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return parser.text()


class ScraperTool:
    """
    Web page fetching tool for agents.

    Supports:
    - ``"simple"``: HTTP fetch via httpx (timeout, redirects, size cap), HTML converted to text
    - ``"playwright"``: browser navigation via LangChain's Playwright tools
    """

    def __init__(
        self,
        provider: str = "simple",
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_chars: int = DEFAULT_MAX_CHARS,
        user_agent: str = DEFAULT_USER_AGENT,
        **kwargs: Any,
    ):
        """
        Initialize scraper tool.

        Args:
            provider: Scraping provider ("simple", "playwright")
            api_key: Reserved for future hosted providers (unused)
            timeout: HTTP timeout in seconds for the "simple" provider
            max_chars: Maximum characters of page text returned
            user_agent: User-Agent header sent by the "simple" provider
            **kwargs: Additional configuration
        """
        if provider not in _PROVIDERS:
            raise ValueError(f"Unknown provider: {provider!r}. Supported: {', '.join(_PROVIDERS)}")
        self.name = "scraper"
        self.description = "Fetch a web page by URL and return its readable text."
        self.provider = provider
        self.api_key = api_key
        self.timeout = timeout
        self.max_chars = max_chars
        self.user_agent = user_agent
        self.config = kwargs
        self._navigate_tool: Any | None = None
        self._initialize_tool()

    def _initialize_tool(self) -> None:
        """Initialize the underlying LangChain tool (playwright provider only)."""
        if self.provider != "playwright":
            return
        if not LANGCHAIN_AVAILABLE or NavigateTool is None:
            raise ImportError(
                "langchain-community is required for the playwright provider. "
                "Install with: pip install langchain-community"
            )
        try:
            self._navigate_tool = NavigateTool()
        except Exception as e:
            raise ImportError(
                "Playwright tools require playwright. "
                "Install with: pip install playwright && playwright install"
            ) from e

    def fetch(self, url: str) -> str:
        """
        Fetch ``url`` and return its text (raises :class:`ScraperError` on failure).

        Only ``http`` and ``https`` URLs are accepted.
        """
        if not url.startswith(("http://", "https://")):
            raise ScraperError(f"Only http(s) URLs are supported, got: {url!r}")

        if self.provider == "playwright" and self._navigate_tool is not None:
            try:
                content = str(self._navigate_tool.invoke({"url": url}))
            except Exception as e:
                raise ScraperError(f"Error scraping {url}: {e}") from e
            return content[: self.max_chars]

        try:
            response = httpx.get(
                url,
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": self.user_agent},
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise ScraperError(f"Error scraping {url}: {e}") from e

        content_type = response.headers.get("content-type", "")
        body = response.text
        text = html_to_text(body) if "html" in content_type or "<html" in body[:500] else body
        return text[: self.max_chars]

    def run(self, url: str, question: str | None = None, **kwargs: Any) -> str:
        """
        Scrape a URL. Errors are returned as text so an agent loop can recover.

        Args:
            url: URL to scrape
            question: Optional question; the page text is returned prefixed by it so the
                calling LLM can answer from the content
            **kwargs: Additional parameters

        Returns:
            Page text, or an ``"Error scraping ..."`` message
        """
        try:
            content = self.fetch(url)
        except ScraperError as e:
            logger.warning("%s", e)
            return str(e)
        if question:
            return f"Question: {question}\nSource: {url}\n\n{content}"
        return content

    def __call__(self, url: str, question: str | None = None, **kwargs: Any) -> str:
        """Make tool callable"""
        return self.run(url, question, **kwargs)

    def as_langchain_tool(self) -> Any | None:
        """
        Return the underlying LangChain tool for direct use (playwright provider only).

        Returns:
            LangChain BaseTool instance or None
        """
        return self._navigate_tool
