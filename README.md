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

Requirements: Linux x86_64, Python 3.12, an NVIDIA GPU with EGL, the Hugging Face
CLI (`hf`), and `ffmpeg` for rendering. Serving Qwen3.8-27B in bf16 needs about
60 GB of free GPU memory.

The harness runs on top of the Castle simulator runtime, which currently lives in a
separate repository that requires access.

```bash
bash scripts/setup_castle.sh ./castle-runtime
export CASTLE_ROOT="$PWD/castle-runtime"
export PYTHONPATH="$CASTLE_ROOT/.runtime-active:$CASTLE_ROOT:$PWD"
bash scripts/download_tasks.sh
```

## Running

Serve the model with vLLM:

```bash
bash scripts/serve_qwen.sh
```

Then run a setting:

```bash
bash scripts/run_scripted.sh        # sanity check with true targets
bash scripts/run_level1.sh          # positions mode
bash scripts/run_level1_worked.sh   # positions mode + worked example
bash scripts/run_level2.sh          # image-only
```

`BASE_URL`, `MODEL`, `DEVICE` and `OUT_ROOT` override the defaults.

Render a saved episode to video:

```bash
bash scripts/render.sh TRAJECTORY ATTEMPT_DIR EPISODE_JSONL OUTPUT_MP4 [MIDDLE_PNG]
```

## Tests

```bash
"$CASTLE_ROOT/.venv/bin/python" -m pytest -q tests/test_harness.py
```

## Known limitations

* `run_python` runs in isolated mode with a minimal environment, but it is not a
  sandbox: the model's code can still read absolute paths on the filesystem.
* One episode per task.

The results in `results/` were produced by the code on the `main` branch. This
`dev` branch fixes issues found afterwards: only one `move_part` runs per turn, the
`get_task` description matches the image-only setting, `run_python` no longer
inherits the harness environment, episode logs store only each turn's new messages
with relative image paths, and temporary episode folders are removed.
