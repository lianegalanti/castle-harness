"""One-world Castle simulator wrapper using the repository's planner and controller."""

from __future__ import annotations

import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from robot_assembly import (
    active_control,
    batched_collect,
    cuboids,
    gpu,
    process_stages,
    tasks,
    world_slots,
)
from scipy.spatial.transform import Rotation

from castle_harness.target_images import render_target_views


class _OneWorldGroup(batched_collect.ConstructionGroup):
    """Keep Castle's batched stage controller on a single live world."""

    def restore(self, world: int, qpos: np.ndarray, qvel: np.ndarray) -> None:
        self.collector.reset(qpos=qpos[None], qvel=qvel[None])
        self.snapshot = self.collector.engine.snapshot()
        self.last[world] = None
        self.rollback_count += 1


class _TrajectoryBuffer:
    def __init__(self, stride: int):
        self.stride = stride
        self.qpos: list[np.ndarray] = []
        self.qvel: list[np.ndarray] = []
        self.segments: list[dict[str, Any]] = []

    @property
    def last_index(self) -> int:
        return len(self.qpos) - 1

    def append(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        positions = np.asarray(qpos, dtype=np.float32)
        velocities = np.asarray(qvel, dtype=np.float32)
        if positions.ndim != 1 or velocities.ndim != 1:
            raise ValueError("Trajectory states must be one-dimensional world states.")
        self.qpos.append(positions.copy())
        self.qvel.append(velocities.copy())

    def append_episode(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        positions = np.asarray(qpos)
        velocities = np.asarray(qvel)
        if positions.ndim == 3:
            positions = positions[:, 0]
        if velocities.ndim == 3:
            velocities = velocities[:, 0]
        if (
            positions.ndim != 2
            or velocities.ndim != 2
            or len(positions) != len(velocities)
            or not len(positions)
        ):
            raise ValueError("Expected aligned episode qpos/qvel state arrays.")
        indices = list(range(0, len(positions), self.stride))
        if indices[-1] != len(positions) - 1:
            indices.append(len(positions) - 1)
        for index in indices:
            self.append(positions[index], velocities[index])

    def finish_segment(
        self,
        start: int,
        tool: str,
        part: int | None,
        args: dict[str, Any],
        result: dict[str, Any],
        qpos: np.ndarray,
        qvel: np.ndarray,
    ) -> None:
        if self.last_index <= start:
            self.append(qpos, qvel)
        ok = bool(result.get("ok", result.get("passed", False)))
        reason = result.get("reason")
        if reason is None and not ok:
            checks = result.get("failing_checks", [])
            reason = ",".join(map(str, checks)) or (
                "fatal" if result.get("fatal") else None
            )
        self.segments.append(
            {
                "tool": tool,
                "part": part,
                "args": args,
                "ok": ok,
                "reason": reason,
                "start": int(start),
                "end": self.last_index,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "qpos": np.stack(self.qpos).astype(np.float32, copy=False),
            "qvel": np.stack(self.qvel).astype(np.float32, copy=False),
            "segments": list(self.segments),
        }


class CastleEnv:
    """Expose checked one-part moves over one live MJ-Warp world."""

    def __init__(
        self,
        attempt_dir: str | Path,
        device: str = "cuda:0",
        record_trajectory: bool = False,
        trajectory_stride: int = 5,
        target_mode: str = "positions",
    ):
        if trajectory_stride < 1:
            raise ValueError("trajectory_stride must be positive")
        if target_mode not in {"positions", "image"}:
            raise ValueError("target_mode must be positions or image")
        self.attempt_dir = Path(attempt_dir).resolve()
        self.device = device
        self.record_trajectory = bool(record_trajectory)
        self.trajectory_stride = int(trajectory_stride)
        self.target_mode = target_mode
        self.target_image_paths: dict[str, Path] = {}
        self._trajectory = (
            _TrajectoryBuffer(self.trajectory_stride)
            if self.record_trajectory
            else None
        )
        self.plan = json.loads((self.attempt_dir / "plan.json").read_text())
        config_path = self.attempt_dir / "execution_config.json"
        self.config = (
            json.loads(config_path.read_text()) if config_path.is_file() else {}
        )
        self.active_part_count = int(
            self.plan.get("active_part_count", len(self.plan["sizes"]))
        )
        order_path = self.attempt_dir / "execution_order.json"
        order = json.loads(order_path.read_text()) if order_path.is_file() else {}
        self.execution_order = order.get(
            "assembly", self.plan.get("assembly_order", [])
        )
        self.model = mujoco.MjModel.from_binary_path(
            str(self.attempt_dir / "model.mjb")
        )
        with np.load(self.attempt_dir / "replay.npz") as replay:
            self.start_qpos = replay["qpos"][0].copy()
            self.start_qvel = replay["qvel"][0].copy()
            self.demo_final_qpos = replay["qpos"][-1].copy()
        self._output = Path(tempfile.mkdtemp(prefix="castle-env-"))
        self.pool = world_slots.ModelPool([self.model], [0], device=device)
        self.collector = gpu.make_collector(
            self.model,
            np.asarray(self.plan["target"], dtype=float),
            1,
            1,
            int(self.config.get("seed", 0)),
            device=device,
            native_models=self.pool.native_models,
            capacity_models=[self.model],
        )
        self.collector.engine.model = self.pool.model
        self.pool.model.block_dim.linesearch_iterative = int(
            self.config.get("line_search_threads", 128)
        )
        self.collector.engine.native_models = self.pool.native_models
        self.collector.controller = active_control.ActivePathControl(self.collector)
        self.group = _OneWorldGroup(
            self.collector,
            float("inf"),
            [self.attempt_dir / "model.mjb"],
        )
        self.group.max_rollbacks = int(self.config.get("max_rollbacks", 0))
        self.group.rollback_count = 0
        self.group.planner_start_method = self.config.get(
            "planner_start_method", "spawn"
        )
        self.group.planning_watchdog_seconds = float(
            self.config.get("planning_watchdog_seconds", 120)
        )
        context = process_stages.planner_context(self.group.planner_start_method)
        planning_workers = int(self.config.get("planning_workers", 1))
        self.group.planning_permits = context.BoundedSemaphore(planning_workers)
        self.sim_steps = 0
        self.started = time.perf_counter()
        self.runner: batched_collect.Runner | None = None
        self.reset()

    @property
    def wall_s(self) -> float:
        """Return elapsed environment wall time for the current episode."""
        return time.perf_counter() - self.started

    def _new_runner(self) -> None:
        self.runner = batched_collect.Runner(
            self.group,
            0,
            self.plan,
            self._output,
            planning_seconds=float(self.config.get("planning_seconds", 8.0)),
            deadline=float("inf"),
            seed=int(self.config.get("seed", 0)),
            shuffle_grasps=True,
        )
        self.group.runners = [self.runner]

    def reset(self) -> dict[str, Any]:
        """Restore the recorded parked state and reset episode accounting."""
        self.collector.reset(
            qpos=self.start_qpos[None].copy(),
            qvel=self.start_qvel[None].copy(),
        )
        self.sim_steps = 0
        self.started = time.perf_counter()
        self._new_runner()
        if self._trajectory is not None:
            qpos, qvel = self._snapshot()
            if not self._trajectory.qpos:
                self._trajectory.append(qpos, qvel)
            else:
                start = self._trajectory.last_index
                self._trajectory.append(qpos, qvel)
                self._trajectory.finish_segment(
                    start,
                    "reset_scene",
                    None,
                    {},
                    {"ok": True},
                    qpos,
                    qvel,
                )
        return self.observe()

    def _snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        state = self.collector.engine.snapshot()
        return state["qpos"][0].copy(), state["qvel"][0].copy()

    def _healthy(self) -> bool:
        try:
            self.collector.engine.assert_healthy()
            qpos, qvel = self._snapshot()
        except Exception:  # noqa: BLE001
            return False
        return bool(np.isfinite(qpos).all() and np.isfinite(qvel).all())

    @staticmethod
    def _yaw_deg(quaternion_wxyz: np.ndarray) -> float:
        quat = np.asarray(quaternion_wxyz, dtype=float)
        rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        return float(rotation.as_euler("zyx", degrees=True)[0])

    def _part_state(self, part: int, qpos: np.ndarray) -> dict[str, Any]:
        pose = qpos[14 + 7 * part : 21 + 7 * part]
        size = np.asarray(self.plan["sizes"][part], dtype=float)
        angle = float(
            np.rad2deg(cuboids.orientation_error(pose[None, 3:], size[None])[0])
        )
        return {
            "index": part,
            "size_m": size.tolist(),
            "position": pose[:3].tolist(),
            "yaw_deg": self._yaw_deg(pose[3:]),
            "upright": bool(angle <= 5.0),
            "orientation_error_deg": angle,
        }

    def observe(self) -> dict[str, Any]:
        """Return compact live state for all active construction parts."""
        qpos, _ = self._snapshot()
        closure = float(qpos[6] / 0.8)
        return {
            "parts": [
                self._part_state(part, qpos) for part in range(self.active_part_count)
            ],
            "arm_at_home": bool(np.max(np.abs(qpos[:6] - tasks.HOME)) <= 0.04),
            "gripper_state": {
                "closure": closure,
                "open": bool(np.max(np.abs(qpos[6:14])) <= 0.08),
            },
            "sim_steps": self.sim_steps,
            "wall_s": self.wall_s,
        }

    def task_spec(self) -> dict[str, Any]:
        """Return the unprivileged target description shown to an agent."""
        parts = []
        for part in range(self.active_part_count):
            spec = {
                "index": part,
                "size_m": list(map(float, self.plan["sizes"][part])),
                "target_yaw_deg": 0.0,
            }
            if self.target_mode == "positions":
                spec["target_position"] = list(map(float, self.plan["target"][part]))
            parts.append(spec)
        task = {"parts": parts}
        if self.target_mode == "image":
            task["images"] = ["top", "front", "side"]
        return task

    def render_target_images(self, output_dir: str | Path) -> dict[str, Path]:
        """Render orthographic views from the accepted demo's final state."""
        if self.target_mode != "image":
            raise ValueError("target images are only available in image mode")
        self.target_image_paths = render_target_views(
            self.demo_final_qpos,
            self.plan["sizes"],
            output_dir,
            self.active_part_count,
        )
        return self.target_image_paths

    def _run_stage(self, name, goals, order, stage_plan):
        assert self.runner is not None
        original_plan = self.runner.plan
        steps_before = self.group.control_steps
        self.runner.plan = stage_plan
        try:
            result = self.group.run_stage(
                name,
                [goals],
                enabled=[True],
                orders=[order],
            )[0]
            if result is None:
                raise RuntimeError("one-world stage returned no result")
            check, episode = result
            used = int(check.get("recorded_control_steps", len(episode["action"])))
            self.sim_steps += used
            if self._trajectory is not None:
                self._trajectory.append_episode(episode["qpos"], episode["qvel"])
            return check, episode, used
        except Exception:
            self.sim_steps += self.group.control_steps - steps_before
            raise
        finally:
            self.runner.plan = original_plan

    def move_part(
        self,
        part: int,
        x: float,
        y: float,
        z: float,
        yaw_deg: float = 0.0,
    ) -> dict[str, Any]:
        if self._trajectory is None:
            return self._move_part_impl(part, x, y, z, yaw_deg)
        start = self._trajectory.last_index
        args = {
            "part": part,
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "yaw_deg": float(yaw_deg),
        }
        try:
            result = self._move_part_impl(part, x, y, z, yaw_deg)
        except Exception as error:  # noqa: BLE001
            result = {
                "ok": False,
                "fatal": not self._healthy(),
                "part": part,
                "reason": f"{type(error).__name__}: {error}",
            }
        try:
            qpos, qvel = self._snapshot()
        except Exception:  # noqa: BLE001
            qpos, qvel = self._trajectory.qpos[-1], self._trajectory.qvel[-1]
        self._trajectory.finish_segment(
            start, "move_part", part, args, result, qpos, qvel
        )
        return result

    def _move_part_impl(
        self,
        part: int,
        x: float,
        y: float,
        z: float,
        yaw_deg: float = 0.0,
    ) -> dict[str, Any]:
        """Plan and execute one checked pick/place without raising normal failures."""
        started = time.perf_counter()
        destination = np.asarray([x, y, z], dtype=float)
        result: dict[str, Any] = {"ok": False, "fatal": False, "part": part}
        if not isinstance(part, int) or part < 0 or part >= self.active_part_count:
            return {**result, "reason": "invalid_part", "sim_steps": 0}
        if not np.isfinite(np.r_[destination, yaw_deg]).all():
            return {**result, "reason": "nonfinite_request", "sim_steps": 0}
        size = np.asarray(self.plan["sizes"][part], dtype=float)
        symmetry = 90.0 if math.isclose(size[0], size[1]) else 180.0
        yaw_error = abs(((float(yaw_deg) + symmetry / 2) % symmetry) - symmetry / 2)
        if yaw_error > 1e-6:
            return {
                **result,
                "reason": "unsupported_yaw",
                "detail": "Castle's planner accepts upright center goals only.",
                "sim_steps": 0,
            }
        assert self.runner is not None
        used = 0
        steps_before = self.sim_steps
        try:
            qpos, _ = self._snapshot()
            current = qpos[14:].reshape(-1, 7)
            goals = np.asarray(self.plan["target"], dtype=float).copy()
            parking = np.asarray(self.plan["parking"], dtype=float).copy()
            goals[: self.active_part_count] = current[: self.active_part_count, :3]
            parking[: self.active_part_count] = current[: self.active_part_count, :3]
            goals[part] = destination
            stage_plan = dict(self.plan)
            stage_plan["parking"] = parking
            stage_plan["fixed_parts"] = sorted(
                set(self.plan.get("fixed_parts", ()))
                | (set(range(self.active_part_count)) - {part})
            )
            check, _, used = self._run_stage(
                f"move_{self.sim_steps}_{part}",
                goals,
                [part],
                stage_plan,
            )
            if check.get("passed"):
                result.update(ok=True, reason=None)
            else:
                reason = check.get("reason")
                if reason is None:
                    failed = [
                        key
                        for key in ("pose_valid", "settled", "dynamics")
                        if check.get(key) is False
                        or (
                            isinstance(check.get(key), dict)
                            and check[key].get("passed") is False
                        )
                    ]
                    reason = "stage_check_failed"
                    if failed:
                        reason += ":" + ",".join(failed)
                result.update(reason=str(reason))
        except Exception as error:  # noqa: BLE001
            used = self.sim_steps - steps_before
            result.update(reason=f"{type(error).__name__}: {error}", sim_steps=used)
            if not self._healthy():
                result["fatal"] = True
        try:
            qpos, _ = self._snapshot()
            final = self._part_state(part, qpos)
            position_error = float(
                np.linalg.norm(np.asarray(final["position"]) - destination)
            )
            result.update(
                final_pose=final,
                error={
                    "position_m": position_error,
                    "yaw_deg": abs(float(final["yaw_deg"]) - float(yaw_deg)),
                },
            )
        except Exception as error:  # noqa: BLE001
            result.update(
                ok=False,
                fatal=True,
                reason=f"state_read_error:{type(error).__name__}: {error}",
            )
        result.update(sim_steps=used, wall_s=time.perf_counter() - started)
        if not self._healthy():
            result.update(ok=False, fatal=True, reason="nonfinite_or_broken_sim")
        return result

    def step(self, action7: list[float] | np.ndarray) -> dict[str, Any]:
        if self._trajectory is None:
            return self._step_impl(action7)
        start = self._trajectory.last_index
        try:
            action_args: Any = np.asarray(action7, dtype=float).tolist()
        except (TypeError, ValueError):
            action_args = str(action7)
        try:
            result = self._step_impl(action7)
        except Exception as error:  # noqa: BLE001
            result = {
                "ok": False,
                "fatal": not self._healthy(),
                "reason": f"{type(error).__name__}: {error}",
            }
        if "qpos" in result and "qvel" in result:
            qpos = np.asarray(result["qpos"], dtype=np.float32)
            qvel = np.asarray(result["qvel"], dtype=np.float32)
        else:
            try:
                qpos, qvel = self._snapshot()
            except Exception:  # noqa: BLE001
                qpos, qvel = self._trajectory.qpos[-1], self._trajectory.qvel[-1]
        self._trajectory.finish_segment(
            start, "step", None, {"action": action_args}, result, qpos, qvel
        )
        return result

    def _step_impl(self, action7: list[float] | np.ndarray) -> dict[str, Any]:
        """Apply one normalized 20 ms seven-dimensional controller command."""
        action = np.asarray(action7, dtype=np.float32)
        if action.shape != (7,) or not np.isfinite(action).all():
            return {"ok": False, "fatal": False, "reason": "invalid_action"}
        try:
            trace, _ = gpu.run_controls(
                self.collector.controller,
                1,
                mode="replay",
                actions=action[None, None, :],
                grip_efforts=np.ones(1, dtype=np.float32),
            )
            self.sim_steps += 1
        except Exception as error:  # noqa: BLE001
            return {
                "ok": False,
                "fatal": not self._healthy(),
                "reason": f"{type(error).__name__}: {error}",
            }
        return {
            "ok": True,
            "fatal": False,
            "qpos": trace["qpos"][-1, 0].tolist(),
            "qvel": trace["qvel"][-1, 0].tolist(),
            "sim_steps": self.sim_steps,
        }

    def _per_part_errors(
        self, qpos: np.ndarray, goals: np.ndarray, sizes: np.ndarray
    ) -> list[dict[str, Any]]:
        poses = np.asarray(qpos)[14:].reshape(-1, 7)
        errors = []
        for part in range(self.active_part_count):
            position_error = float(np.linalg.norm(poses[part, :3] - goals[part]))
            orientation_error = float(
                np.rad2deg(
                    cuboids.orientation_error(
                        poses[part : part + 1, 3:],
                        sizes[part : part + 1],
                    )[0]
                )
            )
            errors.append(
                {
                    "index": part,
                    "position_error_m": position_error,
                    "orientation_error_deg": orientation_error,
                    "upright": orientation_error <= 5.0,
                    "correct": position_error <= 0.005 and orientation_error <= 5.0,
                }
            )
        return errors

    def check(self) -> dict[str, Any]:
        if self._trajectory is None:
            return self._check_impl()
        start = self._trajectory.last_index
        try:
            result = self._check_impl()
        except Exception as error:  # noqa: BLE001
            result = {
                "passed": False,
                "fatal": not self._healthy(),
                "failing_checks": [f"checker_error:{type(error).__name__}:{error}"],
            }
        try:
            qpos, qvel = self._snapshot()
        except Exception:  # noqa: BLE001
            qpos, qvel = self._trajectory.qpos[-1], self._trajectory.qvel[-1]
        self._trajectory.finish_segment(start, "submit", None, {}, result, qpos, qvel)
        return result

    def _check_impl(self) -> dict[str, Any]:
        """Run the repository's final-target hold and support checks."""
        assert self.runner is not None
        started = time.perf_counter()
        failures: list[str] = []
        goals = np.asarray(self.plan["target"], dtype=float)
        sizes = np.asarray(self.plan["sizes"], dtype=float)
        try:
            qpos, _ = self._snapshot()
            parking = np.asarray(self.plan["parking"], dtype=float).copy()
            current = qpos[14:].reshape(-1, 7)
            parking[: self.active_part_count] = current[: self.active_part_count, :3]
            stage_plan = dict(self.plan)
            stage_plan["parking"] = parking
            stage_plan["fixed_parts"] = sorted(
                set(self.plan.get("fixed_parts", ()))
                | set(range(self.active_part_count))
            )
            check, episode, _ = self._run_stage(
                f"terminal_check_{self.sim_steps}",
                goals,
                [],
                stage_plan,
            )
            qpos = episode["qpos"]
            hold = {
                key: check[key]
                for key in (
                    "position_error_m",
                    "orientation_error_deg",
                    "drift_m",
                    "surface_speed_m_s",
                    "home_error_rad",
                    "gripper_error_rad",
                    "pose_valid",
                    "settled",
                    "passed",
                )
                if key in check
            }
            contacts = cuboids.contact_report(self.model, qpos, self.plan["supports"])
            check["passed"] &= contacts["passed"]
        except Exception as error:  # noqa: BLE001
            try:
                current_qpos, _ = self._snapshot()
                per_part = self._per_part_errors(current_qpos, goals, sizes)
            except Exception:  # noqa: BLE001
                per_part = []
            return {
                "passed": False,
                "fatal": not self._healthy(),
                "parts_correct": sum(item["correct"] for item in per_part),
                "per_part_errors": per_part,
                "failing_checks": [f"checker_error:{type(error).__name__}:{error}"],
                "sim_steps": self.sim_steps,
                "wall_s": self.wall_s,
            }
        if not hold.get("pose_valid", False):
            failures.append("pose_valid")
        if not hold.get("settled", False):
            failures.append("settled")
        failures.extend(contacts["failures"])
        if check.get("reason") is not None:
            failures.append(str(check["reason"]))
        per_part = self._per_part_errors(qpos[-1], goals, sizes)
        passed = bool(
            check.get("reason") is None and hold.get("passed") and contacts["passed"]
        )
        return {
            "passed": passed,
            "fatal": not self._healthy(),
            "parts_correct": sum(item["correct"] for item in per_part),
            "per_part_errors": per_part,
            "failing_checks": failures,
            "hold": hold,
            "contacts": contacts,
            "sim_steps": self.sim_steps,
            "wall_s": self.wall_s,
            "check_wall_s": time.perf_counter() - started,
            "goal_field": "plan.json[target]",
            "support_field": "plan.json[supports]",
        }

    def trajectory(self) -> dict[str, Any]:
        """Return recorded world-0 trajectory data, or empty arrays if disabled."""
        if self._trajectory is None:
            return {
                "qpos": np.empty((0, self.model.nq), dtype=np.float32),
                "qvel": np.empty((0, self.model.nv), dtype=np.float32),
                "segments": [],
            }
        return self._trajectory.as_dict()
