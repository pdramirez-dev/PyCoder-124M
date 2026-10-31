#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
exec uv run python scripts/train.py --model configs/model_124m.yaml --config configs/train_124m.yaml --device mps "$@"
