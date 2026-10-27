#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-$ROOT/runs}"
OUT="$OUT_ROOT/out_qwen38_keep_ml_worked"
if [[ -e "$OUT" ]]; then
  printf 'Output already exists: %s\n' "$OUT" >&2
  exit 1
fi

# Reconstructed from the captured worked-process prefix; the stored argv was
# truncated after --request-time and did not preserve a complete command.
exec python "$ROOT/run.py" \
  --backend openai \
  --base-url "${BASE_URL:-http://127.0.0.1:8104/v1}" \
  --model "${MODEL:-Qwen/Qwen3.8-27B}" \
  --tasks U0002,U0007,U0011,U0012,U0024,U0052 \
  --condition keep --order-seed 0 --out "$OUT" \
  --device "${DEVICE:-cuda:0}" \
  --max-turns 300 --max-total-tokens 4000000 \
  --max-sim-steps 2000000 --max-wall-s 10800 \
  --request-timeout 600 --record-trajectory --trajectory-stride 5 \
  --system-prompt prompts/system_level1_worked.txt
