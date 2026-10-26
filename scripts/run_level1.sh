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

run_arm() {
  local name="$1" condition="$2" tasks="$3"
  local out="$OUT_ROOT/$name"
  if [[ -e "$out" ]]; then
    printf 'Output already exists: %s\n' "$out" >&2
    exit 1
  fi
  "$PYTHON" "$ROOT/run.py" \
    --backend openai --base-url "$BASE_URL" --model "$MODEL" \
    --tasks "$tasks" --condition "$condition" --order-seed 0 \
    --out "$out" --device "$DEVICE" \
    --max-turns 300 --max-total-tokens 4000000 \
    --max-sim-steps 2000000 --max-wall-s 10800 \
    --request-timeout 600 --record-trajectory --trajectory-stride 5 \
    --system-prompt prompts/system_level1.txt
}

# The five single-task wipe invocations below match recovered argv. The easy
# grouped run and U0024 keep run use reconstructed argv because their full
# original argv was not retained.
run_arm out_qwen38_keep_easy_rec keep "U0008,U0080,U0149"
for task in U0002 U0007 U0011 U0012 U0052; do
  run_arm "out_qwen38_rec_${task}" wipe "$task"
done
run_arm out_qwen38_keep_ml_rec keep U0024
