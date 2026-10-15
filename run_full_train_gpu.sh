#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Explicit inputs are required; never train on the bundled software fixture.
exec "${PYTHON_BIN:-python}" "$ROOT_DIR/scripts/train_stage.py" --action train "$@"
