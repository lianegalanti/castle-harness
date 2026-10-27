#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 4 || $# -gt 5 ]]; then
  printf 'Usage: %s TRAJECTORY ATTEMPT_DIR EPISODE_LOG OUTPUT_MP4 [STILL_PNG]\n' \
    "$0" >&2
  exit 2
fi

trajectory="$1"
attempt_dir="$2"
episode_log="$3"
output="$4"
still="${5:-}"
still_args=()
if [[ -n "$still" ]]; then
  still_args=(--still-out "$still")
fi

# Reconstructed HD settings from the approved high-resolution renders.
exec python "$ROOT/render_episode.py" \
  --trajectory "$trajectory" \
  --model "$attempt_dir/model.mjb" \
  --attempt-dir "$attempt_dir" \
  --episode-log "$episode_log" \
  --out "$output" \
  --no-text-panel --view-index 1 \
  --view-width 1280 --view-height 960 --target-size 960 \
  "${still_args[@]}"
