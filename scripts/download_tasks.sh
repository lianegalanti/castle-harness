#!/usr/bin/env bash
set -euo pipefail

: "${CASTLE_ROOT:?Set CASTLE_ROOT to the Castle runtime checkout first}"

DATASET="alexmzhang/castle-reasoning"
ARTIFACTS="$CASTLE_ROOT/artifacts"
CAMPAIGN="campaign-tasks-20261011"
declare -a INCLUDE=(
  "README.md"
  "SHA256SUMS"
  "$CAMPAIGN/index.json"
  "$CAMPAIGN/campaign.json"
  "$CAMPAIGN/shards/2ms-b0-000/run00/attempt00216/**"
  "$CAMPAIGN/shards/2ms-b0-000/run00/attempt00217/**"
  "$CAMPAIGN/shards/2ms-b0-000/run00/attempt00219/**"
  "$CAMPAIGN/shards/1ms-b0-000/run00/attempt00007/**"
  "$CAMPAIGN/shards/2ms-b2-000/run00/attempt00105/**"
  "$CAMPAIGN/shards/2ms-b2-000/run00/attempt00041/**"
  "$CAMPAIGN/shards/2ms-b2-000/run00/attempt00020/**"
  "$CAMPAIGN/shards/2ms-b2-000/run00/attempt00025/**"
  "$CAMPAIGN/shards/1ms-b0-000/run00/attempt00000/**"
  "$CAMPAIGN/shards/1ms-b0-000/run00/attempt00046/**"
)

mkdir -p "$ARTIFACTS"
hf download "$DATASET" \
  --repo-type dataset \
  --local-dir "$ARTIFACTS" \
  --include "${INCLUDE[@]}"

cd "$ARTIFACTS"
if [[ ! -f SHA256SUMS ]]; then
  printf 'Downloaded SHA256SUMS is missing\n' >&2
  exit 1
fi

# --ignore-missing lets this manifest cover a selected subset of the public
# dataset while still failing on every downloaded file that it does list.
sha256sum --check --ignore-missing SHA256SUMS
if ! grep -q "$CAMPAIGN/" SHA256SUMS; then
  printf 'WARNING: SHA256SUMS has no %s entries; listed files were checked only.\n' \
    "$CAMPAIGN" >&2
fi

printf 'Downloaded selected Castle task artifacts under %s\n' "$ARTIFACTS"
