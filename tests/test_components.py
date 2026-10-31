"""Unit tests for graphs, routing, planning, validation, RAG, scraping, and LLM plumbing."""

import sqlite3

import httpx
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agentensemble import (
    HybridAgent,
    PlannerAgent,
    RAGAgent,
    RAGTool,
    RouterAgent,
    ScraperTool,
    StateGraphAgent,
    ValidationTool,
    WorkflowGraph,
    estimate_cost,
)
from agentensemble.llm.interface import LLMMessage, LLMResponse, ToolCall
from agentensemble.llm.langchain_base import (
    _extract_usage,
    _messages_to_langchain,
    _parse_ai_message,
)
from agentensemble.memory import SQLiteSession
from agentensemble.tools import scraper as scraper_mod
from agentensemble.tools.adapters import get_tool_schemas, invoke_tool
from agentensemble.tools.scraper import ScraperError, html_to_text
from tests.fakes import ScriptedLLM


class EchoAgent:
    def __init__(self, name):
        self.name = name

    def run(self, query, **kwargs):
        return {"result": f"{self.name}: {query}", "metadata": {}}

    async def arun(self, query, **kwargs):
        return self.run(query, **kwargs)


# --------------------------------------------------------------------------- graphs


class TestWorkflowGraph:
    def test_conditional_routing_and_visited_nodes(self):
        graph = WorkflowGraph(entry="classify")
        graph.add_node("classify", lambda s: {"context": {"urgent": "refund" in s["query"]}})
        graph.add_node("escalate", lambda s: {"result": "escalated"})
        graph.add_node("self_serve", lambda s: {"result": "see FAQ"})
        graph.add_edge("classify", "escalate", lambda s: s["context"]["urgent"])
        graph.add_edge("classify", "self_serve")
        graph.add_edge("escalate", "end")
        graph.add_edge("self_serve", "end")

        urgent = graph.run("I want a refund")
        assert urgent["result"] == "escalated"
        assert urgent["metadata"]["nodes_visited"] == ["classify", "escalate"]

        calm = graph.run("How do I reset my password?")
        assert calm["result"] == "see FAQ"
        assert calm["metadata"]["nodes_visited"] == ["classify", "self_serve"]

    def test_max_steps_bounds_cycles_and_none_returning_nodes(self):
        graph = WorkflowGraph(entry="loop", max_steps=5)
        graph.add_node("loop", lambda s: None)
        graph.add_edge("loop", "loop")
        out = graph.run("q")
        assert out["metadata"]["steps"] == 5

    def test_string_node_jumps(self):
        graph = WorkflowGraph(
            nodes={"start": "finish", "finish": lambda s: {"result": "done"}},
            exit_nodes=["stop"],
        )
        graph.add_edge("finish", "stop")
        assert graph.run("q")["result"] == "done"


def test_stategraph_reports_only_visited_nodes():
    agent = StateGraphAgent(
        nodes={
            "start": lambda s: {"context": {"seen": True}},
            "analyze": lambda s: {"result": "analyzed"},
            "unused": lambda s: {"result": "never"},
        }
    )
    out = agent.run("q")
    assert out["result"] == "analyzed"
    assert out["metadata"]["nodes_visited"] == ["start", "analyze"]


# --------------------------------------------------------------------------- agents


class TestRAGAgent:
    def test_fallback_attempts_are_counted(self):
        class FlakyRag:
            name = "rag"

            def __init__(self):
                self.queries = []

            def run(self, query, urls=None):
                self.queries.append((query, urls))
                return "found it" if len(self.queries) == 3 else "not found"

        rag = FlakyRag()
        out = RAGAgent(tools=[rag]).run("q", urls=["https://example.com"])
        assert out["result"] == "found it"
        assert out["metadata"]["fallback_attempts"] == 2
        # URLs are indexed once, by the primary query only.
        assert [u for _, u in rag.queries] == [["https://example.com"], None, None]


class TestHybridAgent:
    def test_fixed_pipeline_uses_tools(self, monkeypatch):
        for var in ("OPENAI_API_KEY", "MISTRAL_API_KEY", "AGENTENSEMBLE_LLM_PROVIDER"):
            monkeypatch.delenv(var, raising=False)

        class Search:
            name = "search"

            def run(self, query):
                return {"urls": ["https://a.example"]}

        class Rag:
            name = "rag"

            def run(self, query, urls=None):
                return f"context from {urls}"

        agent = HybridAgent(tools=[Search(), Rag(), ValidationTool()])
        out = agent.run("q")
        assert out["metadata"]["actions_taken"] == ["SEARCH", "RAG", "VALIDATE", "ANSWER"]
        assert out["result"] == "context from ['https://a.example']"

    def test_llm_routing_invalid_action_means_answer(self):
        llm = ScriptedLLM(
            [
                LLMResponse(
                    tool_calls=[ToolCall(name="choose_next_action", arguments={"action": "x"})]
                )
            ]
        )
        out = HybridAgent(llm=llm).run("q")
        assert out["metadata"]["actions_taken"] == ["ANSWER"]


class TestRouterAgent:
    def test_routes_with_llm_choice(self):
        llm = ScriptedLLM(
            [
                LLMResponse(
                    tool_calls=[
                        ToolCall(name="route_to_agent", arguments={"agent_name": "billing"})
                    ]
                )
            ]
        )
        router = RouterAgent(
            agents={"tech": EchoAgent("tech"), "billing": EchoAgent("billing")}, llm=llm
        )
        out = router.run("refund please")
        assert out["result"] == "billing: refund please"
        assert out["metadata"]["routed_to"] == "billing"
        # The routing schema restricts choices to known agents.
        schema = llm.calls[0]["tools"][0]
        assert schema.parameters["properties"]["agent_name"]["enum"] == ["tech", "billing"]

    def test_unknown_choice_falls_back_to_first_agent(self):
        llm = ScriptedLLM(
            [
                LLMResponse(
                    tool_calls=[ToolCall(name="route_to_agent", arguments={"agent_name": "?"})]
                )
            ]
        )
        router = RouterAgent(
            agents={"tech": EchoAgent("tech"), "billing": EchoAgent("billing")}, llm=llm
        )
        assert router.run("q")["metadata"]["routed_to"] == "tech"

    @pytest.mark.asyncio
    async def test_route_only(self):
        llm = ScriptedLLM(
            [
                LLMResponse(
                    tool_calls=[
                        ToolCall(name="route_to_agent", arguments={"agent_name": "billing"})
                    ]
                )
            ]
        )
        router = RouterAgent(
            agents={"tech": EchoAgent("tech"), "billing": EchoAgent("billing")}, llm=llm
        )
        assert await router.route_only("invoice") == "billing"

    def test_single_agent_skips_llm(self):
        llm = ScriptedLLM()
        router = RouterAgent(agents={"only": EchoAgent("only")}, llm=llm)
        assert router.run("q")["metadata"]["routed_to"] == "only"
        assert llm.calls == []


class TestPlannerAgent:
    def test_compare_task_is_decomposed_and_executed(self):
        out = PlannerAgent(executor=EchoAgent("exec")).run("compare A and B")
        assert out["metadata"]["results_count"] == 3
        assert out["result"].count("exec:") == 3

    def test_without_executor_only_plans(self):
        out = PlannerAgent().run("research topic X")
        assert out["metadata"]["executed"] == 0
        assert len(out["metadata"]["subtasks"]) == 2


# --------------------------------------------------------------------------- tools


class TestValidationTool:
    def test_rules(self):
        v = ValidationTool(
            min_length=3, max_length=20, required_pattern=r"\d", reject_pattern="error"
        )
        assert v.run("order 42")["valid"] is True
        assert v.run("")["valid"] is False
        assert v.run("no digits here")["valid"] is False
        assert v.run("error 500")["valid"] is False
        assert v.run("x" * 30)["valid"] is False

    def test_deep_mode(self):
        assert ValidationTool(validation_mode="deep").run("n/a")["valid"] is False
        assert ValidationTool(validation_mode="deep").run("a meaningful answer")["valid"] is True


class TestRAGTool:
    @pytest.fixture
    def rag(self):
        tool = RAGTool(embeddings=DeterministicFakeEmbedding(size=64), chunk_size=200)
        tool.index_texts(
            ["Refunds are issued within 14 days.", "Shipping takes 3-5 business days."],
            metadatas=[{"source": "refunds.md"}, {"source": "shipping.md"}],
        )
        return tool

    def test_index_and_retrieve(self, rag):
        context, docs = rag.retrieve("Refunds are issued within 14 days.", k=1)
        assert docs[0].metadata["source"] == "refunds.md"
        assert "14 days" in context

    def test_run_and_retrieval_tool(self, rag):
        assert "Source:" in rag.run("shipping")
        lc_tool = rag.create_retrieval_tool()
        assert lc_tool.name == "retrieve_context"

    def test_metadata_length_mismatch(self):
        tool = RAGTool(embeddings=DeterministicFakeEmbedding(size=8))
        with pytest.raises(ValueError):
            tool.index_texts(["a", "b"], metadatas=[{}])

    def test_index_documents_skips_bad_urls(self, monkeypatch, caplog):
        from agentensemble.tools import rag as rag_mod

        seen = {}

        class FakeLoader:
            def __init__(self, web_paths, requests_kwargs, raise_for_status):
                seen["timeout"] = requests_kwargs["timeout"]
                self.url = web_paths[0]

            def load(self):
                if "bad" in self.url:
                    raise OSError("unreachable")
                return [rag_mod.Document(page_content="Hello world", metadata={"source": self.url})]

        monkeypatch.setattr(rag_mod, "WebBaseLoader", FakeLoader)
        tool = RAGTool(embeddings=DeterministicFakeEmbedding(size=8), fetch_timeout=5)
        tool.index_documents(["https://bad.example", "https://good.example"])
        assert seen["timeout"] == 5
        assert "Failed to load https://bad.example" in caplog.text
        _, docs = tool.retrieve("Hello", k=1)
        assert docs[0].metadata["source"] == "https://good.example"

    def test_default_embeddings_need_a_key(self, monkeypatch):
        for var in ("OPENAI_API_KEY", "MISTRAL_API_KEY", "AGENTENSEMBLE_LLM_PROVIDER"):
            monkeypatch.delenv(var, raising=False)
        with pytest.raises(ValueError, match="No embeddings provider"):
            RAGTool().index_texts(["x"])


class TestScraperTool:
    def test_html_to_text_drops_scripts_and_keeps_blocks(self):
        html = (
            "<html><head><title>t</title><script>evil()</script></head>"
            "<body><h1>Hi</h1><p>One &amp; two</p></body></html>"
        )
        assert html_to_text(html) == "Hi\nOne & two"

    def test_fetch_uses_timeout_and_strips_html(self, monkeypatch):
        captured = {}

        def fake_get(url, **kwargs):
            captured.update(kwargs)
            return httpx.Response(
                200,
                text="<html><body><p>Pricing: $10</p></body></html>",
                headers={"content-type": "text/html"},
                request=httpx.Request("GET", url),
            )

        monkeypatch.setattr(scraper_mod.httpx, "get", fake_get)
        tool = ScraperTool(timeout=3)
        assert tool.run("https://example.com") == "Pricing: $10"
        assert captured["timeout"] == 3
        assert captured["follow_redirects"] is True
        assert tool.run("https://example.com", question="price?").startswith("Question: price?")

    def test_http_errors_become_messages(self, monkeypatch):
        def fake_get(url, **kwargs):
            return httpx.Response(404, request=httpx.Request("GET", url))

        monkeypatch.setattr(scraper_mod.httpx, "get", fake_get)
        tool = ScraperTool()
        assert tool.run("https://example.com/missing").startswith("Error scraping")
        with pytest.raises(ScraperError):
            tool.fetch("https://example.com/missing")

    def test_rejects_non_http_urls_and_unknown_provider(self):
        with pytest.raises(ScraperError):
            ScraperTool().fetch("file:///etc/passwd")
        with pytest.raises(ValueError):
            ScraperTool(provider="nope")

    def test_schema_and_argument_normalisation(self, monkeypatch):
        tool = ScraperTool()
        (schema,) = get_tool_schemas([tool])
        assert schema.parameters["required"] == ["url"]
        monkeypatch.setattr(tool, "fetch", lambda url: f"fetched {url}")
        assert (
            invoke_tool(tool, "scraper", {"query": "https://x.example"})
            == "fetched https://x.example"
        )


# --------------------------------------------------------------------------- memory


def test_sqlite_session_closes_connections(tmp_path, monkeypatch):
    opened = []
    real_connect = sqlite3.connect

    class TrackingConnection(sqlite3.Connection):
        def close(self):
            opened.remove(self)
            super().close()

    def connect(path, *args, **kwargs):
        conn = real_connect(path, *args, factory=TrackingConnection, **kwargs)
        opened.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", connect)
    session = SQLiteSession("s", db_path=str(tmp_path / "s.db"))
    session.add_messages([{"role": "user", "content": "hi"}])
    assert session.get_messages() == [{"role": "user", "content": "hi"}]
    session.clear()
    assert opened == []


# --------------------------------------------------------------------------- llm plumbing


class TestLangChainConversion:
    def test_messages_round_trip(self):
        msgs = _messages_to_langchain(
            [
                LLMMessage(role="system", content="s"),
                LLMMessage(role="user", content="u"),
                LLMMessage(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCall(id="1", name="t", arguments={"a": 1})],
                ),
                LLMMessage(role="tool", content="r", tool_call_id="1"),
            ]
        )
        assert [type(m) for m in msgs] == [SystemMessage, HumanMessage, AIMessage, ToolMessage]
        assert msgs[2].tool_calls[0]["name"] == "t"
        assert msgs[2].tool_calls[0]["type"] == "tool_call"

    def test_parse_ai_message_with_tool_calls_and_usage(self):
        msg = AIMessage(
            content="",
            tool_calls=[{"id": "1", "name": "t", "args": {"q": "x"}}],
            usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        )
        parsed = _parse_ai_message(msg)
        assert parsed.finish_reason == "tool_calls"
        assert parsed.tool_calls[0].arguments == {"q": "x"}
        assert parsed.usage == {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}

    def test_usage_from_response_metadata(self):
        msg = AIMessage(
            content="hi",
            response_metadata={"token_usage": {"prompt_tokens": 7, "completion_tokens": 1}},
        )
        assert _extract_usage(msg) == {"input_tokens": 7, "output_tokens": 1, "total_tokens": 0}
        assert _parse_ai_message(msg).content == "hi"


def test_estimate_cost_prefix_match_and_fallback():
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
    assert estimate_cost(usage, model="gpt-4.1-mini-2025-04-14") == pytest.approx(2.0)
    assert estimate_cost(usage, model="gpt-4.1") == pytest.approx(10.0)
    assert estimate_cost(usage, model="unknown-model") == pytest.approx(8.0)  # mistral-large
    assert estimate_cost(usage, model="m", pricing={"m": (1.0, 1.0)}) == pytest.approx(2.0)


def test_builtin_tools_describe_themselves_to_the_llm():
    schemas = {s.name: s for s in get_tool_schemas([RAGTool(description="Company FAQ")])}
    assert schemas["rag"].description == "Company FAQ"
    assert "knowledge base" in RAGTool().description
    from agentensemble import SearchTool

    (search_schema,) = get_tool_schemas([SearchTool(provider="duckduckgo")])
    assert search_schema.description.startswith("Search the web")


def test_workflow_node_without_outgoing_edge_ends_the_run():
    graph = (
        WorkflowGraph(entry="classify")
        .add_node("classify", lambda s: {"context": {"refund": True}})
        .add_node("refund", lambda s: {"result": "refunds team"})
        .add_edge("classify", "refund", lambda s: s["context"]["refund"])
    )
    out = graph.run("refund please")
    assert out["result"] == "refunds team"
    assert out["metadata"]["nodes_visited"] == ["classify", "refund"]
