# Contributing to siirl-agentic

Thank you for your interest in contributing to siirl-agentic! This document provides guidelines and instructions for contributing.

## Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/sii-research/siirl-agentic.git
cd siirl-agentic
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Development Dependencies

```bash
# Install the package in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Development Workflow

### Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting (line length: 100)
- **isort**: Import sorting
- **Ruff**: Fast Python linter
- **mypy**: Static type checking

All these tools are configured in `pyproject.toml` and run automatically via pre-commit hooks.

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_specific.py

# Run with coverage
pytest --cov=siirl --cov-report=html

# Run specific test markers
pytest -m unit
pytest -m integration
```

### Pre-commit Hooks

Before committing, pre-commit hooks will automatically:
- Format code with Black
- Sort imports with isort
- Check for linting issues with Ruff
- Check spelling with codespell
- Validate YAML files
- Check for trailing whitespace

To run manually:

```bash
pre-commit run --all-files
```

### Type Checking

```bash
mypy siirl
```

## Pull Request Process

1. **Fork the repository** and create your branch from `main`
2. **Write tests** for your changes
3. **Update documentation** if needed
4. **Ensure all tests pass** and pre-commit hooks succeed
5. **Write a clear commit message** describing your changes
6. **Submit a pull request** with a detailed description

### Commit Message Guidelines

- Use clear and descriptive commit messages
- Start with a verb in present tense (e.g., "Add", "Fix", "Update")
- Reference issue numbers when applicable

Example:
```
Add async multi-turn agent support (#123)

- Implement async agent executor
- Add test cases for multi-turn scenarios
- Update documentation
```

## Code Review

All submissions require review. We use GitHub pull requests for this purpose. Reviewers will check:

- Code quality and style
- Test coverage
- Documentation updates
- Performance implications
- Breaking changes

## Testing Guidelines

- Write unit tests for new functionality
- Ensure existing tests pass
- Maintain or improve code coverage
- Use appropriate test markers (unit, integration, e2e)

## Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions/classes
- Update API documentation if needed
- Include examples for new features

## Questions?

If you have questions, please:
- Open an issue on GitHub
- Contact the maintainers
- Check existing documentation

Thank you for contributing to siirl-agentic!
