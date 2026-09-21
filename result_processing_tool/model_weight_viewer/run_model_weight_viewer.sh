#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
HOST="${MODEL_WEIGHT_VIEWER_HOST:-127.0.0.1}"
PORT="${1:-43817}"

if [ -n "${MODEL_WEIGHT_VIEWER_PYTHON:-}" ]; then
    PYTHON_BIN="${MODEL_WEIGHT_VIEWER_PYTHON}"
elif [ -x "/home/tyd/miniconda3/envs/torch/bin/python" ]; then
    PYTHON_BIN="/home/tyd/miniconda3/envs/torch/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
else
    echo "Could not find python3 or python." >&2
    exit 1
fi

LINK="http://${HOST}:${PORT}"
echo "LLR2 model weight viewer: ${LINK}"
echo "Press Ctrl+C to stop."
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/model_weight_viewer.py" --host "${HOST}" --port "${PORT}" --quiet
