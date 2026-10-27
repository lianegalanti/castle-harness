#!/usr/bin/env bash
set -euo pipefail

# argparse uses the last --tasks value, so explicit flags override this default.
exec python -m castle_reasoning.cli download \
  --tasks "${TASKS:-$(python -c 'from castle_harness.tasks import DEFAULT_TASKS; print(DEFAULT_TASKS)')}" \
  "$@"
