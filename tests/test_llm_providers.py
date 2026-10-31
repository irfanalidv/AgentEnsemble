"""Tests for provider selection and the OpenAI/Mistral providers (no network calls)."""

import pytest

from agentensemble.llm import (
    MistralLLMProvider,
    OpenAILLMProvider,
    ToolSchema,
    get_default_provider,
    resolve_provider_name,
)
from agentensemble.llm.openai_provider import DEFAULT_OPENAI_MODEL

ENV_VARS = (
    "AGENTENSEMBLE_LLM_PROVIDER",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_BASE_URL",
    "MISTRAL_API_KEY",
    "MISTRAL_MODEL",
)

TOOL = ToolSchema(
    name="lookup",
    description="Look something up",
    parameters={"type": "object", "properties": {"q": {"type": "string"}}},
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _bound_kwargs(runnable):
    return getattr(runnable, "kwargs", {})


class TestProviderSelection:
    def test_no_keys_resolves_to_none(self):
        assert resolve_provider_name() is None

    def test_no_keys_raises_clear_error(self):
        with pytest.raises(ValueError, match="OPENAI_API_KEY or MISTRAL_API_KEY"):
            get_default_provider()

    def test_openai_preferred_when_both_keys_set(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("MISTRAL_API_KEY", "m-test")
        assert resolve_provider_name() == "openai"
        assert isinstance(get_default_provider(), OpenAILLMProvider)

    def test_mistral_when_only_mistral_key(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "m-test")
        assert isinstance(get_default_provider(), MistralLLMProvider)

    def test_explicit_env_override(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("MISTRAL_API_KEY", "m-test")
        monkeypatch.setenv("AGENTENSEMBLE_LLM_PROVIDER", "mistral")
        assert isinstance(get_default_provider(), MistralLLMProvider)

    def test_provider_argument_wins(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("MISTRAL_API_KEY", "m-test")
        assert isinstance(get_default_provider(provider="mistral"), MistralLLMProvider)

    def test_unknown_provider(self, monkeypatch):
        monkeypatch.setenv("AGENTENSEMBLE_LLM_PROVIDER", "nope")
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_default_provider()


class TestOpenAIProvider:
    def test_requires_key(self):
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAILLMProvider()

    def test_default_model_and_env_override(self, monkeypatch):
        assert OpenAILLMProvider(api_key="sk-test").model_name == DEFAULT_OPENAI_MODEL
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
        assert OpenAILLMProvider(api_key="sk-test").model_name == "gpt-4o-mini"

    def test_temperature_passed_for_chat_models(self):
        provider = OpenAILLMProvider(model="gpt-4.1-mini", temperature=0.3, api_key="sk-test")
        assert provider._model.temperature == 0.3

    @pytest.mark.parametrize("model", ["gpt-5-mini", "o3-mini"])
    def test_temperature_omitted_for_reasoning_models(self, model):
        provider = OpenAILLMProvider(model=model, temperature=0.3, api_key="sk-test")
        assert provider._model.temperature in (None, 1, 1.0)

    def test_base_url(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
        provider = OpenAILLMProvider(api_key="sk-test")
        assert provider._model.openai_api_base == "http://localhost:1234/v1"

    def test_required_tool_choice(self):
        bound = OpenAILLMProvider(api_key="sk-test")._bind_tools([TOOL], "required")
        assert _bound_kwargs(bound)["tool_choice"] == "required"
        assert _bound_kwargs(bound)["tools"][0]["function"]["name"] == "lookup"

    def test_no_tools_returns_model(self):
        provider = OpenAILLMProvider(api_key="sk-test")
        assert provider._bind_tools(None) is provider._model


class TestMistralProvider:
    def test_required_maps_to_any(self):
        bound = MistralLLMProvider(api_key="m-test")._bind_tools([TOOL], "required")
        assert _bound_kwargs(bound)["tool_choice"] == "any"

    def test_auto_sends_no_tool_choice(self):
        bound = MistralLLMProvider(api_key="m-test")._bind_tools([TOOL], "auto")
        assert "tool_choice" not in _bound_kwargs(bound)


def test_agents_pick_up_openai_from_env(monkeypatch):
    from agentensemble import ReActAgent

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert isinstance(ReActAgent(tools=[]).llm, OpenAILLMProvider)


def test_agents_construct_without_any_key():
    from agentensemble import ReActAgent

    assert ReActAgent(tools=[]).llm is None


class TestEventLoopSafety:
    """Sync agent APIs use asyncio.run() per call; async clients must not leak across loops."""

    @staticmethod
    async def _model_after_ensure(provider):
        provider._ensure_model_for_running_loop()
        return provider._model

    @pytest.mark.parametrize("cls,key", [(OpenAILLMProvider, "sk-test"), (MistralLLMProvider, "m")])
    def test_model_rebuilt_on_new_loop(self, cls, key):
        import asyncio

        provider = cls(api_key=key)
        first = asyncio.run(self._model_after_ensure(provider))
        second = asyncio.run(self._model_after_ensure(provider))
        assert first is not second

    def test_model_reused_within_same_loop(self):
        import asyncio

        provider = OpenAILLMProvider(api_key="sk-test")

        async def twice():
            return await self._model_after_ensure(provider), await self._model_after_ensure(
                provider
            )

        a, b = asyncio.run(twice())
        assert a is b

    def test_openai_models_do_not_share_async_client(self):
        p1 = OpenAILLMProvider(api_key="sk-test")
        p2 = OpenAILLMProvider(api_key="sk-test")
        assert p1._model.http_async_client is not p2._model.http_async_client
