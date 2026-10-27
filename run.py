"""Run Castle harness episodes and write one normalized result row per task."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np

import castle_reasoning.data.paths as paths

from castle_harness.agent import Budgets, run_episode
from castle_harness.backends import (
    AnthropicBackend,
    OpenAIBackend,
    ScriptedBackend,
)
from castle_harness.env import CastleEnv
from castle_harness.tasks import DEFAULT_TASKS

CASTLE_ROOT = paths.dataset_root()

INDEXES = (
    [Path(os.environ["CASTLE_INDEX"]).expanduser()]
    if "CASTLE_INDEX" in os.environ
    else paths.dataset_indexes(CASTLE_ROOT)
)


def failure_category(episode: dict) -> str:
    """Classify failures with outcome errors before physical check diagnostics."""
    if episode.get("budget_exhausted"):
        return f"budget_exhausted:{episode['budget_exhausted']}"
    if episode.get("fatal"):
        return "sim_fatal"
    if episode.get("tool_error"):
        return "tool_error"
    if episode.get("passed"):
        return "none"
    hold = episode.get("hold", {})
    if hold.get("pose_valid") and not hold.get("settled", True):
        return "unstable"
    if any(
        not item.get("upright", True) for item in episode.get("per_part_errors", [])
    ):
        return "knocked_over"
    if any(
        check.startswith("checker_error:")
        for check in episode.get("failing_checks", [])
    ):
        return "tool_error"
    return "wrong_pose"


def result_row(
    target_id: str,
    parts: int,
    order_index: int,
    condition: str,
    backend_name: str,
    model: str,
    episode: dict,
) -> dict:
    """Build the stable results.jsonl schema."""
    return {
        "target_id": target_id,
        "parts": parts,
        "order_index": order_index,
        "condition": condition,
        "backend": backend_name,
        "model": model,
        "passed": bool(episode.get("passed")),
        "parts_correct": int(episode.get("parts_correct", 0)),
        "failure_category": failure_category(episode),
        "input_tokens": int(episode.get("input_tokens", 0)),
        "output_tokens": int(episode.get("output_tokens", 0)),
        "sim_steps": int(episode.get("sim_steps", 0)),
        "wall_s": float(episode.get("wall_s", 0)),
        "n_tool_calls": int(episode.get("n_tool_calls", 0)),
        "tokens_per_part": (
            episode.get("input_tokens", 0) + episode.get("output_tokens", 0)
        )
        / parts,
        "sim_steps_per_part": episode.get("sim_steps", 0) / parts,
        "budget_exhausted": episode.get("budget_exhausted"),
        "failing_checks": episode.get("failing_checks", []),
        "per_part_errors": episode.get("per_part_errors", []),
    }


def resolve_attempts(target_ids: list[str]) -> dict[str, Path]:
    """Resolve requested IDs to locally downloaded accepted attempt directories."""
    resolved = {}
    requested = set(target_ids)
    for index_path in INDEXES:
        if not index_path.is_file():
            continue
        index = json.loads(index_path.read_text())
        for row in index["demonstrations"]:
            target_id = row["target_id"]
            if target_id not in requested or target_id in resolved:
                continue
            path = index_path.parent / row["attempt_directory"]
            if (path / "plan.json").is_file():
                resolved[target_id] = path
    for target_id in target_ids:
        if target_id not in resolved:
            raise FileNotFoundError(f"no downloaded accepted attempt for {target_id}")
    return {target_id: resolved[target_id] for target_id in target_ids}


def make_backend(
    name: str,
    model: str,
    base_url: str | None,
    plan: dict,
    execution_order=None,
    request_timeout: float = 600,
):
    """Construct the selected response backend."""
    if name == "scripted":
        return ScriptedBackend(plan, execution_order)
    if name == "anthropic":
        return AnthropicBackend(model, request_timeout=request_timeout)
    if not base_url:
        raise ValueError("--base-url is required for openai")
    return OpenAIBackend(model, base_url, request_timeout=request_timeout)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        default=DEFAULT_TASKS,
        help=f"Comma-separated task IDs (default paper benchmark: {DEFAULT_TASKS})",
    )
    parser.add_argument("--order-seed", type=int, default=0)
    parser.add_argument("--condition", choices=["keep", "wipe"], default="wipe")
    parser.add_argument(
        "--backend", choices=["scripted", "anthropic", "openai"], required=True
    )
    parser.add_argument("--model", default="")
    parser.add_argument("--base-url")
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument(
        "--system-prompt",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--target-mode", choices=["positions", "image"], default="positions"
    )
    parser.add_argument("--record-trajectory", action="store_true")
    parser.add_argument("--trajectory-stride", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-turns", type=int, default=300)
    parser.add_argument("--max-total-tokens", type=int, default=4_000_000)
    parser.add_argument("--max-sim-steps", type=int, default=2_000_000)
    parser.add_argument("--max-wall-s", type=float, default=10800)
    return parser.parse_args()


def resolve_system_prompt(target_mode: str, system_prompt: Path | None) -> Path:
    if system_prompt is None:
        filename = "system_image.txt" if target_mode == "image" else "system_level1.txt"
        system_prompt = Path("prompts") / filename
    if not system_prompt.is_absolute():
        system_prompt = Path(__file__).resolve().parent / system_prompt
    return system_prompt


def save_trajectory(env, path: Path, attempt_dir: Path, stride: int) -> None:
    """Persist an environment's sampled world-0 states and action segments."""
    trajectory = env.trajectory()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        qpos=trajectory["qpos"],
        qvel=trajectory["qvel"],
        segments=np.asarray(json.dumps(trajectory["segments"], allow_nan=False)),
        attempt_dir=np.asarray(str(attempt_dir)),
        stride=np.asarray(stride, dtype=np.int64),
    )


def archive_partial_episode(episode_path: Path) -> None:
    """Preserve an unfinished episode log before rerunning the same position."""
    if not episode_path.exists():
        return
    partial_path = episode_path.with_name(f"{episode_path.name}.partial")
    suffix = 1
    while partial_path.exists():
        partial_path = episode_path.with_name(f"{episode_path.name}.partial.{suffix}")
        suffix += 1
    episode_path.rename(partial_path)


def main():
    args = parse_args()
    if args.target_mode == "image" and args.backend == "anthropic":
        raise ValueError("--target-mode image requires the openai or scripted backend")
    system_prompt_path = resolve_system_prompt(args.target_mode, args.system_prompt)
    target_ids = [value.strip() for value in args.tasks.split(",") if value.strip()]
    random.Random(args.order_seed).shuffle(target_ids)
    attempts = resolve_attempts(target_ids)
    args.out.mkdir(parents=True, exist_ok=True)
    memory_dir = args.out / "memory"
    memory_dir.mkdir(exist_ok=True)
    snapshots = args.out / "memory_snapshots"
    snapshots.mkdir(exist_ok=True)
    budgets = Budgets(
        args.max_turns,
        args.max_total_tokens,
        args.max_sim_steps,
        args.max_wall_s,
    )
    results_path = args.out / "results.jsonl"
    completed_tasks = set()
    if args.resume and results_path.is_file():
        completed_tasks = {
            json.loads(line)["target_id"]
            for line in results_path.read_text().splitlines()
            if line.strip()
        }
    with results_path.open("a") as results:
        for order_index, target_id in enumerate(target_ids):
            if target_id in completed_tasks:
                continue
            if args.condition == "wipe":
                shutil.rmtree(memory_dir)
                memory_dir.mkdir()
            episode_path = (
                args.out / "episodes" / f"{order_index:03d}_{target_id}.jsonl"
            )
            if args.resume:
                archive_partial_episode(episode_path)
            env = CastleEnv(
                attempts[target_id],
                args.device,
                record_trajectory=args.record_trajectory,
                trajectory_stride=args.trajectory_stride,
                target_mode=args.target_mode,
            )
            if args.target_mode == "image":
                env.render_target_images(
                    episode_path.parent / f"{episode_path.stem}_images"
                )
            backend = make_backend(
                args.backend,
                args.model,
                args.base_url,
                env.plan,
                env.execution_order,
                args.request_timeout,
            )
            try:
                episode = run_episode(
                    env,
                    backend,
                    memory_dir,
                    budgets,
                    episode_path,
                    system_prompt_path=system_prompt_path,
                )
            finally:
                if args.record_trajectory:
                    save_trajectory(
                        env,
                        args.out
                        / "episodes"
                        / f"{order_index:03d}_{target_id}.trajectory.npz",
                        attempts[target_id],
                        args.trajectory_stride,
                    )
            row = result_row(
                target_id,
                env.active_part_count,
                order_index,
                args.condition,
                args.backend,
                args.model,
                episode,
            )
            results.write(json.dumps(row, allow_nan=False) + "\n")
            results.flush()
            destination = snapshots / f"{order_index:03d}_{target_id}"
            shutil.copytree(memory_dir, destination, dirs_exist_ok=True)
            print(json.dumps(row, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
