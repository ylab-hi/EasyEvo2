# Contributing to EasyEvo2

Thank you for your interest in contributing to EasyEvo2!

## Reporting Bugs

Please open a [GitHub Issue](https://github.com/ylab-hi/EasyEvo2/issues) with:
- A clear description of the bug
- Steps to reproduce
- Expected vs. actual behavior
- Your environment (OS, Python version, GPU model)

## Suggesting Features

Open an issue with the **feature request** label describing:
- The use case / problem you're solving
- Your proposed solution
- Any relevant references (papers, tools, etc.)

## Development Setup

```bash
# Clone the repository
git clone https://github.com/ylab-hi/EasyEvo2.git
cd EasyEvo2

# Install dependencies (requires uv)
uv sync -U

# Run tests
make test

# Lint and format
make lint
make format
```

## Code Style

This project follows strict coding standards enforced by **Ruff**. See
[AGENTS.md](AGENTS.md) for the full style guide. Key rules:

- **Line length:** 88 characters
- **Docstrings:** NumPy-style
- **Imports:** Absolute only, isort-ordered
- **Type annotations:** Required on all public functions
- **Error messages:** Assign to a variable before raising

## Pull Request Process

1. Fork the repository and create a feature branch from `main`
2. Write tests for any new functionality
3. Ensure `make lint` and `make test` pass with zero errors
4. Submit a PR with a clear description of changes

## License

By contributing, you agree that your contributions will be licensed under the
MIT License.
