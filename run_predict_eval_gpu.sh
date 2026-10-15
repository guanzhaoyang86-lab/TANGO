#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# predict.py writes predictions and the accompanying .metrics.json together.
exec "${PYTHON_BIN:-python}" "$ROOT_DIR/predict.py" --max-new-tokens 2048 --torch-dtype bfloat16 "$@"
