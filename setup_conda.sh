#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="mask_iteration_sam3"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BIN="${CONDA_EXE:-/opt/anaconda3/bin/conda}"
CHECKPOINT_SRC="${SAM3_CHECKPOINT_SRC:-$HOME/Desktop/sam3.pt}"
CHECKPOINT_DST="$PROJECT_DIR/third_party/sam3/checkpoints/sam3.pt"

if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  "$CONDA_BIN" create -y -n "$ENV_NAME" python=3.11
fi

"$CONDA_BIN" run -n "$ENV_NAME" python -m pip install --upgrade pip setuptools wheel
"$CONDA_BIN" run -n "$ENV_NAME" python -m pip install torch torchvision
"$CONDA_BIN" run -n "$ENV_NAME" python -m pip install -r "$PROJECT_DIR/requirements.txt"
"$CONDA_BIN" run -n "$ENV_NAME" python -m pip install -e "$PROJECT_DIR/third_party/sam3"

mkdir -p "$(dirname "$CHECKPOINT_DST")"
if [ ! -e "$CHECKPOINT_DST" ]; then
  if [ ! -e "$CHECKPOINT_SRC" ]; then
    echo "Missing SAM3 checkpoint: $CHECKPOINT_SRC" >&2
    exit 1
  fi
  ln "$CHECKPOINT_SRC" "$CHECKPOINT_DST" 2>/dev/null || cp "$CHECKPOINT_SRC" "$CHECKPOINT_DST"
fi

echo "Conda env ready: $ENV_NAME"
echo "SAM3 checkpoint: $CHECKPOINT_DST"
echo "Run: ./run_conda.sh"
