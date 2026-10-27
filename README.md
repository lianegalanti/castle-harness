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
pixi run --locked download-tasks --tasks S0507,U0008
```

The manifest pins both installable distributions to public benchmark commit
[`4da3f207`](https://github.com/smbml/castle-reasoning-benchmark/tree/4da3f2076990acb858984ae24a8f3609cc105b3a):
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

The runner searches both downloaded indices for the first locally available
accepted attempt per requested target. `CASTLE_INDEX` explicitly selects one
custom index. Download each requested task before running; normal runs, imports,
and tests do not download data or generate new campaigns.

## Running

Run the deterministic reference without a model server:

```bash
TASKS=S0507,U0008 pixi run --locked run-scripted
```

For language-model evaluation, serve the model separately with
`bash scripts/serve_qwen.sh`, then run a setting:

```bash
pixi run --locked run-level1
pixi run --locked run-level1-worked
pixi run --locked run-level2
```

Download every requested task explicitly before running. `BASE_URL`, `MODEL`,
`DEVICE`, and `OUT_ROOT` retain their existing meanings; `TASKS` selects scripted
tasks. The existing `run.py` CLI is also available through `pixi run --locked run`.
The native runtime check validates installation and a MuJoCo step; it does not
execute a GPU episode or measure language-model performance.

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

* `run_python` is not sandboxed.
* Several `move_part` calls in one model turn are all executed.
* One episode per task.
