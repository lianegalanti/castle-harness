#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  set -- --tasks "${TASKS:-U0008}"
fi
exec python -m castle_reasoning.cli download "$@"
