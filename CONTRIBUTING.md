# Contributing to AgentEnsemble

Thank you for your interest in contributing to AgentEnsemble! 🎭

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/<your-username>/AgentEnsemble.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Install development dependencies: `pip install -e ".[dev]"`

## Development Setup

```bash
# Install in development mode (uv or pip)
uv pip install -e ".[dev]"      # or: pip install -e ".[dev]"

# Optional: run ruff and hygiene hooks on every commit
pre-commit install

# The same checks CI runs
ruff check .
ruff format --check .
mypy agentensemble
pytest --cov=agentensemble
```

Tests must not call real LLM or web APIs: use the fakes in `tests/fakes.py` and `monkeypatch`.

## Code Style

- Follow PEP 8
- Use `ruff format` for formatting (line length: 100)
- Use `ruff check` for linting (CI runs both)
- Type hints on public functions; `mypy agentensemble` must pass

## Pull Request Process

1. Update documentation if needed
2. Add tests for new features
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Submit PR with clear description

## Areas for Contribution

- New agent implementations
- Additional orchestration patterns
- Tool integrations
- Test cases and benchmarks
- Documentation improvements
- Bug fixes

Thank you for contributing! 🚀
