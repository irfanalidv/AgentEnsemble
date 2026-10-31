"""Tests for package metadata, ReActAgent with a fake LLM, and SearchTool behaviour."""

import pytest

import agentensemble
from agentensemble import ReActAgent, function_tool
from agentensemble.llm.interface import LLMProvider, LLMResponse, ToolCall
from agentensemble.tools import SearchTool


class FakeLLM(LLMProvider):
    """Deterministic provider: first asks for a tool call, then answers with its result."""

    model_name = "fake-model"

    def __init__(self):
        self.calls = []

    def generate(self, messages, tools=None, tool_choice=None, **kwargs):
        raise NotImplementedError

    async def agenerate(self, messages, tools=None, tool_choice=None, **kwargs):
        self.calls.append({"tools": tools, "messages": list(messages)})
        tool_results = [m for m in messages if m.role == "tool"]
        if not tool_results:
            return LLMResponse(
                tool_calls=[ToolCall(id="c1", name="multiply", arguments={"a": 6, "b": 7})],
                finish_reason="tool_calls",
                usage={"input_tokens": 10, "output_tokens": 5},
            )
        return LLMResponse(content=f"The answer is {tool_results[-1].content}")


@function_tool(description="Multiply two integers")
def multiply(a: int, b: int) -> int:
    return a * b


def test_version_exposed():
    assert isinstance(agentensemble.__version__, str)
    assert agentensemble.__version__.count(".") >= 2


def test_react_agent_with_fake_llm_calls_tool():
    llm = FakeLLM()
    agent = ReActAgent(name="calc", tools=[multiply], llm=llm, max_iterations=3)
    result = agent.run("What is 6 times 7?")
    assert "42" in result["result"]
    assert len(llm.calls) == 2
    # First call offered the tool schema, second call (after tool result) did not
    assert llm.calls[0]["tools"] and llm.calls[0]["tools"][0].name == "multiply"


def test_react_agent_without_key_constructs_but_run_fails_clearly(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    agent = ReActAgent(name="nokey", tools=[])
    assert agent.llm is None
    with pytest.raises(RuntimeError, match="LLM required"):
        agent.run("hello")


def test_search_tool_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        SearchTool(provider="bing")


def test_search_tool_is_lazy_and_uses_backend(monkeypatch):
    tool = SearchTool(provider="duckduckgo")
    assert tool._tool is None  # nothing created until first use

    class StubBackend:
        def invoke(self, query):
            return f"results for {query}"

    monkeypatch.setattr(tool, "_initialize_tool", lambda: setattr(tool, "_tool", StubBackend()))
    assert tool("python") == "results for python"


def test_search_tool_missing_backend_gives_install_hint(monkeypatch):
    tool = SearchTool(provider="duckduckgo")

    def boom():
        raise ImportError("no ddgs")

    monkeypatch.setattr(tool, "_build_tool", boom)
    with pytest.raises(ImportError, match=r"agentensemble\[search\]"):
        tool.run("python")


def test_mistral_model_env_override(monkeypatch):
    from agentensemble.llm.mistral_provider import MistralLLMProvider

    monkeypatch.setenv("MISTRAL_API_KEY", "dummy-key")
    monkeypatch.setenv("MISTRAL_MODEL", "mistral-small-latest")
    assert MistralLLMProvider().model_name == "mistral-small-latest"
    assert MistralLLMProvider(model="open-mistral-nemo").model_name == "open-mistral-nemo"
    monkeypatch.delenv("MISTRAL_MODEL")
    assert MistralLLMProvider().model_name == "mistral-large-latest"
