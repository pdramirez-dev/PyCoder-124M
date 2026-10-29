# Repository Guidelines

## Project Structure & Module Organization

`src/pycoder/` contains the Python package: `data/` handles preprocessing, tokenization and datasets; `model/` implements the decoder; `training/` manages optimization and checkpoints; and `inference/` generates text. Shared configuration lives in `config.py`. Command-line entry points are in `scripts/`, YAML configurations in `configs/`, and pytest tests in `tests/`. `data/demo/` is a small sample corpus, not a production training set.

## Build, Test, and Development Commands

Use Python 3.12 and `uv`:

- `uv sync` installs the project and development dependencies.
- `uv run pytest -q` runs the test suite.
- `uv run ruff check src scripts tests` checks Python style and errors.
- `uv run python scripts/train_tokenizer.py --vocab-size 512` trains a demo tokenizer; follow it with `uv run python scripts/prepare_dataset.py`.
- `uv run python scripts/train.py --config configs/train_smoke.yaml` runs a three-step smoke training job.
- `uv run python scripts/evaluate.py checkpoints/latest.pt` evaluates a checkpoint; `uv run python scripts/generate.py checkpoints/latest.pt --prompt 'def '` generates code.

See the README for full training and resume commands. Avoid launching long training runs without a clear need.

## Coding Style & Naming Conventions

Follow the existing Python style: four-space indentation, `snake_case` for modules, functions and variables, and `PascalCase` for classes. Ruff is configured for a 100-character line length. Keep model, data and training responsibilities in their respective modules and use YAML for experiment settings.

## Testing Guidelines

Tests use pytest and are named `tests/test_*.py`, with test functions named `test_*`. Reuse the `tiny_config` fixture for inexpensive model tests where appropriate. Add focused tests for behavior changes and run the full suite before submitting. No coverage threshold is configured.

## Commits & Pull Requests

Git history is unavailable in this checkout, so no existing commit convention can be verified. Use concise, imperative commit subjects describing the change. Pull requests should explain the motivation and behavior, link relevant issues, and report tests and lint checks run; include configuration or training details when they affect reproducibility.

## Data & Checkpoints

Generated tokenizers, processed data, logs and checkpoints are ignored by Git. Preserve the tokenizer and data split when resuming training, and load checkpoints only from trusted sources. Review corpus licensing and provenance before training on external data.
