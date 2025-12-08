# Contributing to AgentEnsemble

Thank you for your interest in contributing to AgentEnsemble! 🎭

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/agentensemble.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Install development dependencies: `pip install -e ".[dev]"`

## Development Setup

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black agentensemble/

# Lint code
ruff check agentensemble/
```

## Code Style

- Follow PEP 8
- Use `black` for formatting (line length: 100)
- Use `ruff` for linting
- Type hints are encouraged

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

