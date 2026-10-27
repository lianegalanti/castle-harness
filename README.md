# Castle LLM harness

An evaluation harness that lets a language model build Castle block structures by
calling tools. The model decides which part to move, where, and in what order.
Castle's pick-and-place planner does the motion.

## Tools

| Kind | Tools |
|---|---|
| Read | `get_task()`, `observe()` |
| Act | `move_part(part, x, y, z)`, `reset_scene()` |
| Think | `run_python(code)`, `list_memory()`, `read_memory(path)`, `write_memory(path, content)` |
| Finish | `submit()` |

A task passes when every part is within 5 mm and 5° of its target, correctly
supported, and stable for 1 s.

## Settings

* **Positions mode:** `get_task()` gives each part's target position.
* **Positions mode + worked example:** the same, with a solved task in the system prompt.
* **Image-only:** `get_task()` gives part sizes and three orthographic drawings of the
  target (top, front, side) on a metric grid, but no positions.

## Results

Qwen3.8-27B, one episode per task, on 9 tasks (3 easy with 7 parts, 3 medium with
19–24 parts, 3 hard with 29–31 parts).

| Setting | Easy | Medium | Hard | Total |
|---|---|---|---|---|
| Scripted solver (true targets) | 3/3 | 3/3 | 3/3 | 9/9 |
| Positions mode | 3/3 | 3/3 | 3/3 | 9/9 |
| Image-only | 2/3 | 0/3 | 0/3 | 2/9 |

Per-task tables, token counts, timings and failure reasons are in
`results/SUMMARY.md`. Each run folder has a `results.jsonl` and per-episode logs.

## Setup

Use Linux x86_64, [Pixi](https://pixi.sh/), and an NVIDIA GPU with a working
driver and EGL. Pixi installs Python 3.12, the runtime dependencies, and FFmpeg.
Model serving remains a separate environment.

```bash
pixi install --locked
pixi run --locked check-runtime
pixi run --locked download-tasks
```

The manifest pins both installable distributions to public benchmark commit
[`06c65431`](https://github.com/smbml/castle-reasoning-benchmark/tree/06c65431640a96e5eadec37ffeb553279fb51f4b):
`castle-reasoning-benchmark` and the patched `mujoco-warp` package from
`third_party/mujoco-warp`. The committed `pixi.lock` fixes the complete harness
environment. Updating the benchmark requires updating both revisions together
and regenerating the lockfile. No research checkout, submodule, editable install,
or PYTHONPATH configuration is needed.

`CASTLE_ROOT` means downloaded task data. It defaults to the ignored `dataset/`
directory under the invoking directory, created by the download command. Set
`CASTLE_ROOT` or pass `--root` to `download-tasks` to use another location.
Downloads reuse the existing HF artifacts at the benchmark's pinned dataset
revision and verify their source identities and hashes. Both published campaigns
are supported: 570 `S` targets and 502 `U` targets, totaling 1,072 structures.
For targets with multiple demonstrations, downloading selects the first accepted
indexed attempt. Files retain their original campaign paths.
The default download budget is 1 GiB to fit the nine paper tasks; pass
`--max-download-mb` to set another limit.

The runner searches both downloaded indices for the first locally available
accepted attempt per requested target. `CASTLE_INDEX` explicitly selects one
custom index. Download each requested task before running; normal runs, imports,
and tests do not download data or generate new campaigns.

## Running

Omitting `--tasks` selects the paper's nine-task benchmark, all from the
October 11 campaign (`campaign-tasks-20261011`):

| Difficulty | Parts | Task IDs |
| --- | --- | --- |
| Easy | 7 | `U0008`, `U0080`, `U0149` |
| Medium | 19–24 | `U0002`, `U0007`, `U0011` |
| Hard | 29–31 | `U0012`, `U0024`, `U0052` |

The downloader and scripted command use the same default. Pass `--tasks` to
`pixi run --locked run` or `pixi run --locked download-tasks` to select another
set; `TASKS` also overrides the downloader and scripted command defaults.
For example, `TASKS=U0008 pixi run --locked run-scripted` runs a single task.

The other `run.py` defaults also match the paper runs: `--condition wipe` (memory
cleared between tasks) and budgets of 300 turns, 4M tokens, 2M simulation steps
and 3 hours per task.

Run the deterministic reference without a model server:

```bash
pixi run --locked run-scripted
```

For language-model evaluation, serve the model separately with
`bash scripts/serve_qwen.sh`, then run a setting:

```bash
pixi run --locked run-level1
pixi run --locked run-level1-worked
pixi run --locked run-level2
```

Download every requested task explicitly before running. `BASE_URL`, `MODEL`,
`DEVICE`, and `OUT_ROOT` retain their existing meanings. The existing `run.py`
CLI is also available through `pixi run --locked run`.
The runtime check requires a working NVIDIA GPU and validates installation plus
native MuJoCo and CUDA MuJoCo-Warp steps. It does not execute a full robot episode
or measure language-model performance.

Render a saved recording:

```bash
pixi run --locked render TRAJECTORY ATTEMPT_DIR EPISODE_JSONL OUTPUT_MP4 [MIDDLE_PNG]
```

## Tests

```bash
pixi run --locked test
```

Tests cover harness orchestration, installed Git package identities, runtime
coordination, data-root handling, and both campaign indices. End-to-end GPU
validation uses existing HF tasks with the scripted backend, records fresh
`move_part` executions, and checks the complete runtime success predicate. See
[runtime integration](docs/runtime-integration.md) for the package boundary, pin
updates, and a bounded GPU smoke command.

## Known limitations

* `run_python` runs in isolated mode with a minimal environment, but it is not a
  sandbox: the model's code can still read absolute paths on the filesystem.
* One episode per task.

The results in `results/` were produced by the code on the `main` branch, with the
earlier private Castle runtime. This
`dev` branch fixes issues found afterwards: only one `move_part` runs per turn, the
`get_task` description matches the image-only setting, `run_python` no longer
inherits the harness environment, episode logs store only each turn's new messages
with relative image paths, and temporary episode folders are removed.
