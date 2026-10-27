#!/usr/bin/env bash
set -euo pipefail

# Place defaults first so explicit flags override them.
exec python -m castle_reasoning.cli download \
  --tasks "${TASKS:-$(python -c 'from castle_harness.tasks import DEFAULT_TASKS; print(DEFAULT_TASKS)')}" \
  --max-download-mb 1024 \
  "$@"
