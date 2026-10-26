#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${CASTLE_ROOT:?Set CASTLE_ROOT to the Castle runtime checkout first}"
export PYTHONPATH="$CASTLE_ROOT/.runtime-active:$CASTLE_ROOT:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${CASTLE_ROOT}/.venv/bin/python"
BASE_URL="${BASE_URL:-http://127.0.0.1:8104/v1}"
MODEL="${MODEL:-Qwen/Qwen3.8-27B}"
DEVICE="${DEVICE:-cuda:0}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs}"
mkdir -p "$OUT_ROOT"

# Reconstructed: the nine historical image-mode argv lines were not retained.
# Each task is deliberately a separate wipe invocation, matching the runs.
for task in U0008 U0080 U0149 U0002 U0007 U0011 U0012 U0024 U0052; do
  out="$OUT_ROOT/out_qwen38_image/$task"
  if [[ -e "$out" ]]; then
    printf 'Output already exists: %s\n' "$out" >&2
    exit 1
  fi
  "$PYTHON" "$ROOT/run.py" \
    --backend openai --base-url "$BASE_URL" --model "$MODEL" \
    --tasks "$task" --condition wipe --order-seed 0 \
    --target-mode image --out "$out" --device "$DEVICE" \
    --max-turns 300 --max-total-tokens 4000000 \
    --max-sim-steps 2000000 --max-wall-s 10800 \
    --request-timeout 600 --record-trajectory --trajectory-stride 5
done
