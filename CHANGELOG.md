# Changelog

All notable changes to AgentEnsemble will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.1] - 2026-10-31

### Fixed
- `estimate_cost` priced Mistral at outdated rates: Large at $2/$6 per 1M tokens (now
  $0.70/$2.10) and Small at $0.35/$0.56 (now $0.15/$0.60). The table is now checked against
  the official OpenAI, Anthropic and Mistral pricing pages, dated by `PRICING_AS_OF`, and adds
  current GPT-6/5.x/5, o3/o4-mini, Claude 4.5+/5.x and Ministral/Codestral entries.

## [0.6.0] - 2026-10-31

### Fixed
- `ReActAgent` session memory: the user's question and the final answer were never saved, so
  follow-up turns lost context. A turn now stores the question, tool exchanges and answer.
- `ReActAgent` disabled tools on every turn after the first turn that used one (tool messages
  in the history counted as "already called"). Only the current turn is checked now.
- Session history windows could start with an orphan `tool` message, which chat APIs reject.
- `RunConfig(session=...)` had no effect on `ReActAgent`.
- `SQLiteSession` leaked one open connection per call (`with sqlite3.connect()` does not close).
- `WorkflowGraph`: a node without a matching outgoing edge re-ran until the 50-step limit
  instead of ending the run. `WorkflowGraph` and `StateGraphAgent` reported every node as
  visited; `nodes_visited` now lists the nodes that ran.
- `ScraperTool(provider="simple")` returned placeholder text instead of fetching the page.
- `RAGAgent` always reported `fallback_attempts=0` and re-fetched URLs on every fallback.
- `RAGTool` and `SearchTool` had no tool description, so models saw "Tool: rag" and often
  skipped them. URL loading in `RAGTool` had no timeout and failures were printed.
- `HybridAgent` blocked the event loop while running sync tools.
- `import agentensemble` logged a langchain-community USER_AGENT warning.
- LangChain tool calls were built with `type="function"` instead of `"tool_call"`.

### Added
- `ScraperTool` fetches with httpx (timeout, redirects, `max_chars`, http(s) only) and converts
  HTML to text; `ScraperTool.fetch()` raises `ScraperError`. New `html_to_text()` helper.
- `RAGTool.index_texts()` indexes local text without network; `RAGTool(description=...,
  fetch_timeout=...)`.
- `get_default_embeddings()` / `get_openai_embeddings()`: `RAGTool` embeddings follow the same
  OpenAI-first provider selection as chat models.
- `WorkflowGraph(max_steps=...)` and a `steps` count in its metadata.
- `estimate_cost` knows the `gpt-4.1` family and matches dated model names by prefix.
- Four offline-runnable examples (customer-support RAG, ticket triage, order assistant with
  approval, CSV analytics) and a rule-based `LLMProvider` for running examples without a key.
- CodeQL workflow, pre-commit config, mypy and coverage gate (85%) in CI; `httpx` is now a
  declared dependency.

### Changed
- Library code logs through `logging` instead of printing.
- Ruff rules broadened (SIM, C4, RUF, PTH, RET, PIE); mypy runs with `check_untyped_defs`.
- README rewritten: architecture diagrams, configuration and API tables, references.

## [0.5.0] - 2026-10-31

### Added
- **OpenAI support.** New `OpenAILLMProvider` (LangChain `ChatOpenAI`, native tool calling).
  Default model `gpt-4.1-mini`, configurable via `OPENAI_MODEL`; `OPENAI_BASE_URL` for
  OpenAI-compatible endpoints. Temperature is omitted for reasoning models (o-series, gpt-5).
- `get_default_provider()` / `resolve_provider_name()`: agents now pick OpenAI when
  `OPENAI_API_KEY` is set, otherwise Mistral; `AGENTENSEMBLE_LLM_PROVIDER` forces one.
- `agentensemble.utils.get_chat_model()` and `get_openai_model()`; `StructuredAgent` uses the
  same OpenAI-or-Mistral selection.
- `langchain-openai` is now a core dependency.

### Fixed
- Calling a sync agent API twice (e.g. `agent.run()` then `agent.run()`, or `run()` followed by
  `asyncio.run(agent.arun())`) failed with `RuntimeError: Event loop is closed`. Each sync call
  uses `asyncio.run()`, and the LLM's async HTTP client stayed bound to the first, closed loop.
  Providers now rebuild their chat model when used on a new event loop, and each OpenAI model
  gets its own async HTTP client instead of the process-wide cached one.

### Changed
- Shared LangChain conversion logic moved to `agentensemble.llm.langchain_base`
  (`LangChainChatProvider`); `MistralLLMProvider` keeps its public API.

## [0.4.0] - 2026-10-31

### Fixed
- `ReActAgent` can now be constructed without `MISTRAL_API_KEY` (matching `HybridAgent` and
  `RouterAgent`); `run()` / `astream()` raise a clear `RuntimeError` until an LLM is configured.
  Previously the constructor raised `ValueError`, which broke 6 tests and any offline usage.
- `SearchTool`'s DuckDuckGo fallback works again: `langchain-community` now needs the `ddgs`
  package (the `search` extra previously installed the old `duckduckgo-search`).
- `SearchTool` creates its backend lazily on first use, so it can be registered in a
  `ToolRegistry` without optional dependencies; a missing backend raises an `ImportError` with
  the `pip install 'agentensemble[search]'` hint. Unknown providers still fail fast.
- Tool schema type inference uses identity checks (`ann is str`), not equality.

### Added
- `MISTRAL_MODEL` environment variable to choose the default model for `MistralLLMProvider`.
- `__version__` read from installed package metadata (`importlib.metadata`).
- GitHub Actions CI (ruff lint + format check, pytest on Python 3.10–3.13).
- New tests: ReActAgent tool-calling loop with a fake LLM, SearchTool behaviour, version, model
  override (64 tests total).
- SECURITY.md, CODE_OF_CONDUCT.md, issue/PR templates, Dependabot, CITATION.cff.

### Changed
- Requires Python 3.10+ (3.9 is end-of-life); classifiers for 3.10–3.13.
- Packaging is `pyproject.toml`-only (removed `setup.py` and `requirements.txt`), PEP 639
  `license = "MIT"`, LICENSE and CHANGELOG shipped in the sdist.
- Dependency floors raised to the versions the code actually needs: `langchain>=1.1`,
  `langchain-core>=1.0`, `langchain-community>=0.3`, `langchain-mistralai>=0.2`. The unused
  direct `langgraph` dependency was dropped (it is still installed transitively by `langchain`).
- `dev` extra now includes `build` and `twine`; `black` replaced by `ruff format`.
- Codebase formatted with `ruff format` and modernised type hints (`X | None`).

## [0.3.0] - 2026-03-10

### Added
- Observability (`TraceHooks`, token usage, cost estimation), human-in-the-loop
  (`interrupt_before_tools`), architecture diagram.

## [0.2.0] - 2026-03-10

### Added
- `RouterAgent`, `PlannerAgent`, `WorkflowGraph`, tracing.

## [0.1.0] - 2025-12-09

### Added
- Initial release
- ReActAgent implementation
- StateGraphAgent implementation
- RAGAgent with fallback strategies
- HybridAgent with intelligent routing
- Ensemble orchestration patterns (Supervisor, Swarm, Pipeline)
- Tool ecosystem (Search, Scraper, RAG, Validator)
- Tool registry for centralized management
- Testing framework (AgentComparison, Benchmark, Metrics)
- Base agent class and state management
- Comprehensive documentation

### Infrastructure
- Python 3.9+ support
- pyproject.toml for modern packaging
- Development dependencies (pytest, black, ruff, mypy)
- MIT License
- Contributing guidelines
