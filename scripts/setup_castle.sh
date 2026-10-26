#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/smbml/castle-reasoning-benchmark.git"
REPO_BRANCH="robo-alex/campaign-14-results"
REPO_COMMIT="c1686198cd0ee9a8f178baa4a72aa56ad6f022d5"
INSTALL_DIR="${1:-./castle-runtime}"

if [[ -e "$INSTALL_DIR" ]]; then
  printf 'Refusing to overwrite existing install directory: %s\n' "$INSTALL_DIR" >&2
  exit 1
fi

git clone --single-branch --branch "$REPO_BRANCH" "$REPO_URL" "$INSTALL_DIR"
git -C "$INSTALL_DIR" checkout --detach "$REPO_COMMIT"

cd "$INSTALL_DIR"

# The checked-in construction_setup.md at this commit uses a Python venv and
# requirements-robot.txt. .venv (rather than its .venv-robot example name)
# matches the harness verification command and runtime convention.
python3.12 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -r requirements-robot.txt
python -m pip check
export MUJOCO_GL=egl
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python -m scripts.check_construction_install

python -m pip download --no-deps --only-binary=:all: \
  mujoco-warp==3.13.0 --dest .runtime-source/wheels
printf '%s\n' \
  'cb1f74c36684ef5d2b610926d8979db8aa39fd78613a7130cf6a7ef802ee5e3c  .runtime-source/wheels/mujoco_warp-3.13.0-py3-none-any.whl' \
  | sha256sum --check
python -m zipfile -e \
  .runtime-source/wheels/mujoco_warp-3.13.0-py3-none-any.whl \
  .runtime-source/upstream
python -m scripts.build_active_mujoco_warp \
  --source .runtime-source/upstream/mujoco_warp \
  --output .runtime-active
export PYTHONPATH="$PWD/.runtime-active:$PWD"
python -m scripts.check_construction_install --require-active

ROOT="$(pwd)"
printf '\nSetup complete. In each new shell run:\n'
printf 'export CASTLE_ROOT=%q\n' "$ROOT"
printf 'export PYTHONPATH=%q\n' "$ROOT/.runtime-active:$ROOT"
