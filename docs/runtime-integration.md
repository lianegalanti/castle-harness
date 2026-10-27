# Runtime integration

The harness installs two distributions from the same full Git commit of the
[public benchmark](https://github.com/smbml/castle-reasoning-benchmark):
`castle-reasoning-benchmark` from the repository root, and `mujoco-warp` from
`third_party/mujoco-warp`. The patched distribution provides the active-world
stepping API; installing the upstream PyPI package alone is insufficient.
Both revisions are explicit in `pixi.toml`; `pixi.lock` records the complete
resolved environment. The installed-Git regression checks actual package origins,
commit IDs, and the runtime subdirectory.

`pixi run --locked check-runtime` requires a working NVIDIA GPU. It verifies
source/build identity and takes finite native MuJoCo and CUDA MuJoCo-Warp steps;
it exits with an error when CUDA is unavailable. Full robot episodes remain a
separate check.

The harness imports `castle_reasoning.runtime` and creates a policy-free runtime
session. It supplies requested moves to the runtime stage coordinator and uses
its physics, motion planning, and success checks. Dataset path helpers locate
published inputs; the demonstration teacher is not imported. `CASTLE_ROOT` is
input data rather than a source checkout. No import-path hooks are installed.

## Updating the pin

Change both full revisions together, run `pixi install` to regenerate the lock,
then run `pixi run --locked test` and `pixi run --locked check-runtime`. The
benchmark's dependency metadata does not replace the harness's own lockfile.

## Bounded end-to-end smoke

On a machine with an available NVIDIA GPU, download one task from each campaign
and execute the scripted backend without a model server:

```sh
pixi run --locked download-tasks --tasks S0507,U0008
pixi run --locked run --backend scripted --tasks S0507,U0008 \
  --max-turns 30 --max-total-tokens 10000 --max-sim-steps 200000 \
  --max-wall-s 900 --record-trajectory --trajectory-stride 10 \
  --out runs/runtime-smoke
```

The output directory must be fresh. Inspect `results.jsonl`: each task should
have `passed: true`, all parts correct, `failure_category: "none"`, and no budget
exhaustion. A process exit alone does not establish episode success. Episode
logs and sampled trajectories are under `episodes/`; the README's rendering
command uses those recordings and each resolved attempt directory.

This smoke performs fresh robot moves against the existing HF tasks. It checks
installation, task lookup, planning, execution, success evaluation, and recording
for two small structures; it is not a language-model evaluation or coverage of
all 1,072 targets. Keep model serving in its separate environment.
