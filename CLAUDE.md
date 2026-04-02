# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See README.md for project overview and usage documentation.

## Commands

```bash
uv sync              # Install deps (dev deps included by default)
uv run pytest        # Run all tests
uv run pytest tests/test_cli.py::test_init  # Run a single test
uv run ruff check         # Lint
uv run ruff format --check  # Verify formatting (CI runs this)
uv run ruff format        # Auto-fix formatting
uv run pyright            # Type check
```

## Code conventions

- Python 3.12+, `src/autoresearch_lab/` layout with hatchling build
- Config uses dataclasses deserialized with `dacite` (strict mode), not pydantic
- Scores are always lower-is-better
- `templates/` directory is excluded from ruff
- Tests mock subprocess/Docker but use real dataclasses; verdicts simulated via iterators
