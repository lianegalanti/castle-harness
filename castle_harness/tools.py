"""Agent tool schemas, safe memory access, and dispatch."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

MAX_MEMORY_BYTES = 64 * 1024


def _function(name: str, description: str, properties=None, required=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS = [
    _function("get_task", "Get part sizes and final target poses."),
    _function("observe", "Observe the current scene state."),
    _function(
        "move_part",
        "Move one part to an upright target center.",
        {
            "part": {"type": "integer"},
            "x": {"type": "number"},
            "y": {"type": "number"},
            "z": {"type": "number"},
            "yaw_deg": {"type": "number", "default": 0.0},
        },
        ["part", "x", "y", "z"],
    ),
    _function("reset_scene", "Reset all parts to the parked start."),
    _function("submit", "Check the final construction and end the episode."),
    _function(
        "run_python",
        "Run Python for up to 60 seconds in the episode scratch directory.",
        {"code": {"type": "string"}},
        ["code"],
    ),
    _function("list_memory", "List persistent memory files."),
    _function(
        "read_memory",
        "Read one persistent memory file.",
        {"path": {"type": "string"}},
        ["path"],
    ),
    _function(
        "write_memory",
        "Write one persistent memory file, at most 64 KiB.",
        {"path": {"type": "string"}, "content": {"type": "string"}},
        ["path", "content"],
    ),
]


class ToolDispatcher:
    """Bind tool calls to one environment and one isolated memory directory."""

    def __init__(self, env, memory_dir: str | Path, scratch_dir: str | Path):
        self.env = env
        self.memory_dir = Path(memory_dir).resolve()
        self.scratch_dir = Path(scratch_dir).resolve()
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.terminated = False
        self.terminal_result = None
        self.n_calls = 0
        self._write_context(env.observe())

    def _write_context(self, state: dict[str, Any]) -> None:
        (self.scratch_dir / "task.json").write_text(
            json.dumps(self.env.task_spec(), indent=2) + "\n"
        )
        (self.scratch_dir / "state.json").write_text(json.dumps(state, indent=2) + "\n")
        link = self.scratch_dir / "memory"
        if link.is_symlink() or link.exists():
            if link.is_dir() and not link.is_symlink():
                shutil.rmtree(link)
            else:
                link.unlink()
        link.symlink_to(self.memory_dir, target_is_directory=True)

    def _memory_path(self, value: str) -> Path:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("memory path must be relative and may not contain '..'")
        resolved = (self.memory_dir / Path(*path.parts)).resolve()
        if resolved != self.memory_dir and self.memory_dir not in resolved.parents:
            raise ValueError("memory path escapes memory directory")
        return resolved

    def dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute one named tool call and return JSON-compatible data."""
        self.n_calls += 1
        if self.terminated:
            return {"ok": False, "error": "episode_terminated"}
        if name == "get_task":
            result = self.env.task_spec()
        elif name == "observe":
            result = self.env.observe()
        elif name == "move_part":
            result = self.env.move_part(**arguments)
        elif name == "reset_scene":
            result = self.env.reset()
        elif name == "submit":
            result = self.env.check()
            self.terminated = True
            self.terminal_result = result
        elif name == "run_python":
            result = self._run_python(arguments["code"])
        elif name == "list_memory":
            result = {
                "files": sorted(
                    str(path.relative_to(self.memory_dir))
                    for path in self.memory_dir.rglob("*")
                    if path.is_file()
                )
            }
        elif name == "read_memory":
            path = self._memory_path(arguments["path"])
            if path.stat().st_size > MAX_MEMORY_BYTES:
                raise ValueError("memory file exceeds 64 KiB")
            result = {"path": arguments["path"], "content": path.read_text()}
        elif name == "write_memory":
            content = arguments["content"]
            if len(content.encode()) > MAX_MEMORY_BYTES:
                raise ValueError("memory file exceeds 64 KiB")
            path = self._memory_path(arguments["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            result = {"ok": True, "path": arguments["path"]}
        else:
            raise ValueError(f"unknown tool: {name}")
        if name in {"observe", "move_part", "reset_scene"}:
            self._write_context(self.env.observe())
        return result

    def _run_python(self, code: str) -> dict[str, Any]:
        try:
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=self.scratch_dir,
                text=True,
                capture_output=True,
                timeout=60,
                check=False,
            )
            return {
                "returncode": completed.returncode,
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
            }
        except subprocess.TimeoutExpired as error:
            return {
                "returncode": None,
                "timeout": True,
                "stdout": (error.stdout or "")[-4000:],
                "stderr": (error.stderr or "")[-4000:],
            }
