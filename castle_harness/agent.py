"""Budgeted tool-calling episode loop with JSONL turn logs."""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from castle_harness.tools import ToolDispatcher, tool_schemas

ONE_MOVE_PER_TURN_ERROR = (
    "only one move_part runs per turn; call it again after you see this move's result"
)


@dataclasses.dataclass
class Budgets:
    """Episode limits enforced around backend turns and simulator tools."""

    max_turns: int = 60
    max_total_tokens: int = 400_000
    max_sim_steps: int = 200_000
    max_wall_s: float = 1800.0


def _exhausted(
    budgets: Budgets, turns: int, tokens: int, sim_steps: int, wall_s: float
) -> str | None:
    if turns >= budgets.max_turns:
        return "turns"
    if tokens >= budgets.max_total_tokens:
        return "total_tokens"
    if sim_steps >= budgets.max_sim_steps:
        return "sim_steps"
    if wall_s >= budgets.max_wall_s:
        return "wall_s"
    return None


def _messages_for_log(messages: list[dict[str, Any]], log_dir: Path) -> list:
    """Copy messages with image paths made relative to the log directory."""
    logged = copy.deepcopy(messages)
    for message in logged:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = part["image_url"].get("url", "")
                if url.startswith("/"):
                    try:
                        part["image_url"]["url"] = os.path.relpath(url, log_dir)
                    except ValueError:
                        part["image_url"]["url"] = Path(url).name
    return logged


def run_episode(
    env,
    backend,
    memory_dir,
    budgets: Budgets,
    log_path,
    system_prompt_path=None,
):
    """Run until submit or exhaustion, then return check and accounting data."""
    prompt_path = (
        Path(system_prompt_path)
        if system_prompt_path is not None
        else Path(__file__).resolve().parents[1] / "prompts/system_level1.txt"
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompt_path.read_text().strip()},
        {"role": "user", "content": "Begin the task."},
    ]
    scratch = Path(tempfile.mkdtemp(prefix="castle-episode-"))
    dispatcher = ToolDispatcher(env, memory_dir, scratch)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    schemas = tool_schemas(getattr(env, "target_mode", "positions"))
    totals = {"input_tokens": 0, "output_tokens": 0}
    logged_upto = 0
    turns = 0
    tool_error = False
    exhausted = None
    started = time.perf_counter()
    with log_path.open("w") as log:
        set_event_handler = getattr(backend, "set_event_handler", None)

        def log_backend_event(event):
            log.write(json.dumps(event, allow_nan=False) + "\n")
            log.flush()

        while not dispatcher.terminated:
            exhausted = _exhausted(
                budgets,
                turns,
                totals["input_tokens"] + totals["output_tokens"],
                env.sim_steps,
                time.perf_counter() - started,
            )
            if exhausted:
                break
            turn_started = time.perf_counter()
            if set_event_handler is not None:
                set_event_handler(log_backend_event)
            try:
                assistant, usage = backend.respond(messages, schemas)
            finally:
                if set_event_handler is not None:
                    set_event_handler(None)
            turns += 1
            totals["input_tokens"] += int(usage.get("input_tokens", 0))
            totals["output_tokens"] += int(usage.get("output_tokens", 0))
            messages.append({"role": "assistant", **assistant})
            tool_results = []
            requested_images = []
            moved_this_turn = False
            for call in assistant.get("tool_calls", []):
                try:
                    if call["name"] == "move_part" and moved_this_turn:
                        result = {"ok": False, "error": ONE_MOVE_PER_TURN_ERROR}
                    else:
                        result = dispatcher.dispatch(call["name"], call["arguments"])
                        moved_this_turn |= call["name"] == "move_part"
                except Exception as error:  # noqa: BLE001
                    tool_error = True
                    result = {
                        "ok": False,
                        "error": f"{type(error).__name__}: {error}",
                    }
                tool_results.append(
                    {"id": call["id"], "name": call["name"], "result": result}
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(result, allow_nan=False),
                    }
                )
                if (
                    call["name"] == "get_task"
                    and isinstance(result, dict)
                    and result.get("images")
                    and getattr(backend, "supports_images", False)
                ):
                    requested_images.append(result["images"])
                if dispatcher.terminated:
                    break
            image_paths = getattr(env, "target_image_paths", {})
            for image_names in requested_images:
                if not isinstance(image_paths, dict):
                    raise FileNotFoundError("target image paths are unavailable")
                content = [
                    {
                        "type": "text",
                        "text": (
                            "Target images from get_task(): " + ", ".join(image_names)
                        ),
                    }
                ]
                for name in image_names:
                    image_path = image_paths.get(name)
                    if image_path is None:
                        raise FileNotFoundError(f"target image {name!r} is unavailable")
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": str(Path(image_path).resolve())},
                        }
                    )
                messages.append({"role": "user", "content": content})
            # Log only the messages added this turn; earlier ones are in earlier
            # records, so the full history is the concatenation of all records.
            record = {
                "turn": turns,
                "messages": _messages_for_log(messages[logged_upto:], log_path.parent),
                "tool_calls": assistant.get("tool_calls", []),
                "tool_results": tool_results,
                "usage": usage,
                "timings": {
                    "turn_s": time.perf_counter() - turn_started,
                    "episode_s": time.perf_counter() - started,
                },
            }
            logged_upto = len(messages)
            log.write(json.dumps(record, allow_nan=False) + "\n")
            log.flush()
    shutil.rmtree(scratch, ignore_errors=True)
    if dispatcher.terminal_result is None:
        try:
            terminal = env.check()
        except Exception as error:  # noqa: BLE001
            terminal = {
                "passed": False,
                "fatal": True,
                "parts_correct": 0,
                "per_part_errors": [],
                "failing_checks": [f"checker_error:{type(error).__name__}:{error}"],
            }
    else:
        terminal = dispatcher.terminal_result
    return {
        **terminal,
        "budget_exhausted": exhausted,
        "tool_error": tool_error,
        "turns": turns,
        "input_tokens": totals["input_tokens"],
        "output_tokens": totals["output_tokens"],
        "n_tool_calls": dispatcher.n_calls,
        "sim_steps": env.sim_steps,
        "wall_s": time.perf_counter() - started,
    }
