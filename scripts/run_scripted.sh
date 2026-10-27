#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TASKS="${TASKS:-$(python -c 'from castle_harness.tasks import DEFAULT_TASKS; print(DEFAULT_TASKS)')}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs}"
OUT="$OUT_ROOT/scripted_${TASKS//,/_}"
if [[ -e "$OUT" ]]; then
  printf 'Output already exists: %s\n' "$OUT" >&2
  exit 1
fi

# Recovered invocation shape (the output path here is reconstructed).
exec python "$ROOT/run.py" \
  --backend scripted \
  --tasks "$TASKS" \
  --out "$OUT" \
  --device "${DEVICE:-cuda:0}" \
  --max-turns 300 --max-total-tokens 4000000 \
  --max-sim-steps 2000000 --max-wall-s 10800 \
  --request-timeout 600 --record-trajectory --trajectory-stride 5
