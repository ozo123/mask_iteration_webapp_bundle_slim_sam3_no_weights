#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${CONDA_ENV_NAME:-mask_iteration_sam3}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BIN="${CONDA_EXE:-/opt/anaconda3/bin/conda}"

export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

exec "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
  python "$PROJECT_DIR/start_webapp.py" \
  --sam3-repo-dir "$PROJECT_DIR/third_party/sam3" \
  --checkpoint "$PROJECT_DIR/third_party/sam3/checkpoints/sam3.pt" \
  --device auto \
  --validate-tools-dir "$PROJECT_DIR/Validate_tools" \
  "$@"
