# Changelog

All notable changes to AgentEnsemble will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

