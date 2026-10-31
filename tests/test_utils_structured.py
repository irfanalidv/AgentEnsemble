"""Tests for model/embedding helpers, StructuredAgent result handling, and SearchTool fallbacks."""

import logging

import pytest
from pydantic import BaseModel

from agentensemble.agents import structured_agent as sa
from agentensemble.tools import SearchTool
from agentensemble.tools import search as search_mod
from agentensemble.utils import llm as llm_utils

KEYS = ("OPENAI_API_KEY", "MISTRAL_API_KEY", "AGENTENSEMBLE_LLM_PROVIDER", "OPENAI_MODEL")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in KEYS:
        monkeypatch.delenv(var, raising=False)


class TestModelHelpers:
    def test_chat_model_follows_provider_selection(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        monkeypatch.setenv("MISTRAL_API_KEY", "m-test")
        assert type(llm_utils.get_chat_model()).__name__ == "ChatOpenAI"
        assert type(llm_utils.get_chat_model(provider="mistral")).__name__ == "ChatMistralAI"

    def test_chat_model_without_keys(self):
        with pytest.raises(ValueError, match="No LLM provider configured"):
            llm_utils.get_chat_model()
        with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
            llm_utils.get_mistral_api_key()

    def test_default_embeddings(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        emb = llm_utils.get_default_embeddings()
        assert type(emb).__name__ == "OpenAIEmbeddings"
        assert emb.model == "text-embedding-3-small"
        monkeypatch.setenv("MISTRAL_API_KEY", "m-test")
        assert type(llm_utils.get_default_embeddings("mistral")).__name__ == "MistralAIEmbeddings"

    def test_openai_embeddings_need_key(self):
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            llm_utils.get_openai_embeddings()


class Contact(BaseModel):
    name: str
    email: str


class TestStructuredAgent:
    def test_create_agent_path_extracts_structured_response(self, monkeypatch):
        captured = {}

        class FakeCompiledAgent:
            def invoke(self, payload, **kwargs):
                captured["payload"] = payload
                return {
                    "messages": [{"role": "assistant", "content": "Found Ada."}],
                    "structured_response": Contact(name="Ada", email="ada@example.com"),
                }

        def fake_create_agent(**kwargs):
            captured["kwargs"] = kwargs
            return FakeCompiledAgent()

        monkeypatch.setattr(sa, "CREATE_AGENT_AVAILABLE", True)
        monkeypatch.setattr(sa, "create_agent", fake_create_agent)
        agent = sa.StructuredAgent(
            model=object(), response_format=Contact, system_prompt="be terse"
        )
        out = agent.run("Who is Ada?")

        assert out["structured_response"].email == "ada@example.com"
        assert out["result"] == "Found Ada."
        assert out["metadata"]["method"] == "create_agent"
        assert captured["kwargs"]["response_format"] is Contact
        assert captured["kwargs"]["system_prompt"] == "be terse"
        assert captured["payload"]["messages"][0]["content"] == "Who is Ada?"

    def test_with_structured_output_path(self, monkeypatch):
        class FakeModel:
            def with_structured_output(self, schema):
                class Bound:
                    def invoke(self, query):
                        return schema(name="Bob", email="bob@example.com")

                return Bound()

        monkeypatch.setattr(sa, "CREATE_AGENT_AVAILABLE", False)
        agent = sa.StructuredAgent(model=FakeModel(), response_format=Contact)
        out = agent.run("Who is Bob?")
        assert out["structured_response"].name == "Bob"
        assert out["metadata"]["method"] == "with_structured_output"
        assert agent.invoke({"messages": [{"content": "Who is Bob?"}]})["structured_response"]


class TestSearchFallbacks:
    def test_serper_without_key_falls_back_with_warning(self, monkeypatch, caplog):
        monkeypatch.delenv("SERPER_API_KEY", raising=False)

        class FakeDDG:
            def invoke(self, query):
                return f"ddg:{query}"

        monkeypatch.setattr(search_mod, "DuckDuckGoSearchRun", FakeDDG)
        tool = SearchTool(provider="serper")
        tool.api_key = None
        with caplog.at_level(logging.WARNING, logger="agentensemble.tools.search"):
            assert tool.run("python") == "ddg:python"
        assert tool.provider == "duckduckgo"
        assert "SERPER_API_KEY not set" in caplog.text
