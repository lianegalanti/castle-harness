"""CPU-only tests for orchestration, tool safety, and HTTP wire formats."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from urllib.error import HTTPError

import numpy as np
import pytest

import castle_harness.agent as harness_agent
import castle_harness.backends as harness_backends
import run as harness_run
from castle_harness.agent import ONE_MOVE_PER_TURN_ERROR, Budgets, run_episode
from castle_harness.backends import AnthropicBackend, OpenAIBackend, ScriptedBackend
from castle_harness.env import CastleEnv, _TrajectoryBuffer
from castle_harness.target_images import render_target_views
from castle_harness.tools import ToolDispatcher
from render_episode import (
    _add_target_panel,
    _add_text_panel,
    _map_episode_text_to_segments,
    _panel_text_content,
    _segment_caption,
    _select_view,
)
from run import failure_category, result_row


class FakeEnv:
    def __init__(self):
        self.sim_steps = 0

    def observe(self):
        return {"parts": [], "sim_steps": self.sim_steps}

    def task_spec(self):
        return {"parts": [{"index": 0}]}

    def reset(self):
        self.sim_steps = 0
        return self.observe()

    def move_part(self, **arguments):
        self.sim_steps += 3
        return {"ok": True, "arguments": arguments}

    def check(self):
        return {
            "passed": False,
            "fatal": False,
            "parts_correct": 0,
            "per_part_errors": [{"upright": True}],
            "failing_checks": ["pose_valid"],
        }


class EmptyBackend:
    def respond(self, messages, tools):
        self.messages = list(messages)
        return {"content": "thinking", "tool_calls": []}, {
            "input_tokens": 2,
            "output_tokens": 1,
        }


class ImageBackend:
    supports_images = True

    def __init__(self):
        self.requests = []

    def respond(self, messages, _tools):
        self.requests.append(json.loads(json.dumps(messages)))
        if len(self.requests) == 1:
            call = {"id": "get-task", "name": "get_task", "arguments": {}}
        else:
            call = {"id": "submit", "name": "submit", "arguments": {}}
        return {"content": "", "tool_calls": [call]}, {
            "input_tokens": 1,
            "output_tokens": 1,
        }


def test_memory_path_isolation(tmp_path):
    dispatcher = ToolDispatcher(FakeEnv(), tmp_path / "memory", tmp_path / "scratch")
    with pytest.raises(ValueError):
        dispatcher.dispatch("read_memory", {"path": "../escape"})
    with pytest.raises(ValueError):
        dispatcher.dispatch("write_memory", {"path": "/abs", "content": "x"})
    with pytest.raises(ValueError):
        dispatcher.dispatch(
            "write_memory", {"path": "large", "content": "x" * (64 * 1024 + 1)}
        )
    assert dispatcher.dispatch(
        "write_memory", {"path": "notes/a.txt", "content": "safe"}
    )["ok"]
    assert (
        dispatcher.dispatch("read_memory", {"path": "notes/a.txt"})["content"] == "safe"
    )


def test_dispatch_and_python_context(tmp_path):
    env = FakeEnv()
    dispatcher = ToolDispatcher(env, tmp_path / "memory", tmp_path / "scratch")
    moved = dispatcher.dispatch(
        "move_part", {"part": 0, "x": 1, "y": 2, "z": 3, "yaw_deg": 0}
    )
    assert moved["ok"] and env.sim_steps == 3
    output = dispatcher.dispatch(
        "run_python",
        {
            "code": "import json; print(json.load(open('task.json'))['parts'][0]['index'])"
        },
    )
    assert output["returncode"] == 0 and output["stdout"].strip() == "0"
    submitted = dispatcher.dispatch("submit", {})
    assert not submitted["passed"] and dispatcher.terminated


def test_image_task_spec_omits_positions_and_positions_mode_is_unchanged():
    env = CastleEnv.__new__(CastleEnv)
    env.plan = {"sizes": [[0.1, 0.2, 0.3]], "target": [[0.4, 0.5, 0.6]]}
    env.active_part_count = 1
    env.target_mode = "positions"
    assert env.task_spec() == {
        "parts": [
            {
                "index": 0,
                "size_m": [0.1, 0.2, 0.3],
                "target_position": [0.4, 0.5, 0.6],
                "target_yaw_deg": 0.0,
            }
        ]
    }

    env.target_mode = "image"
    image_spec = env.task_spec()
    assert image_spec == {
        "parts": [{"index": 0, "size_m": [0.1, 0.2, 0.3], "target_yaw_deg": 0.0}],
        "images": ["top", "front", "side"],
    }
    assert all("target_position" not in part for part in image_spec["parts"])


def test_image_task_json_omits_goal_positions(tmp_path):
    class ImageTaskEnv(FakeEnv):
        def task_spec(self):
            return {
                "parts": [{"index": 0, "size_m": [0.1, 0.2, 0.3], "target_yaw_deg": 0}],
                "images": ["top", "front", "side"],
            }

    scratch = tmp_path / "scratch"
    dispatcher = ToolDispatcher(ImageTaskEnv(), tmp_path / "memory", scratch)
    task = json.loads((scratch / "task.json").read_text())
    assert task == dispatcher.dispatch("get_task", {})
    assert "target_position" not in json.dumps(task)


def test_get_task_appends_image_references_after_tool_messages(tmp_path):
    env = FakeEnv()
    env.target_image_paths = {
        name: tmp_path / f"{name}.png" for name in ("top", "front", "side")
    }
    env.task_spec = lambda: {
        "parts": [{"index": 0}],
        "images": ["top", "front", "side"],
    }
    backend = ImageBackend()
    run_episode(
        env,
        backend,
        tmp_path / "memory",
        Budgets(max_turns=3),
        tmp_path / "episode.jsonl",
    )

    second_request = backend.requests[1]
    assert [message["role"] for message in second_request][-2:] == ["tool", "user"]
    image_message = second_request[-1]
    assert image_message["content"][0] == {
        "type": "text",
        "text": "Target images from get_task(): top, front, side",
    }
    assert [part["image_url"]["url"] for part in image_message["content"][1:]] == [
        str(env.target_image_paths[name]) for name in ("top", "front", "side")
    ]
    records = [
        json.loads(line)
        for line in (tmp_path / "episode.jsonl").read_text().splitlines()
    ]
    assert records[0]["messages"][-1]["content"][1]["image_url"]["url"] == "top.png"


def test_openai_serializes_image_reference_as_data_url(tmp_path):
    image_path = tmp_path / "top.png"
    image_path.write_bytes(b"png bytes")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "target"},
                {"type": "image_url", "image_url": {"url": str(image_path)}},
            ],
        }
    ]

    serialized = OpenAIBackend._serialize(messages)
    assert serialized[0]["content"][1]["image_url"]["url"] == (
        "data:image/png;base64,cG5nIGJ5dGVz"
    )
    assert messages[0]["content"][1]["image_url"]["url"] == str(image_path)


def test_target_views_render_from_active_demo_parts(tmp_path):
    qpos = np.zeros(14 + 7 * 3, dtype=np.float32)
    qpos[14:17] = [0.4, 0.0, 0.02]
    qpos[17] = 1.0
    qpos[21:24] = [0.4, 0.05, 0.04]
    qpos[24] = 1.0
    qpos[28:31] = [3.0, 3.0, 3.0]
    qpos[31] = 1.0
    sizes = [[0.04, 0.04, 0.04], [0.08, 0.02, 0.02], [0.05, 0.05, 0.05]]
    paths = render_target_views(qpos, sizes, tmp_path / "episode_images", 2)

    assert list(paths) == ["top", "front", "side"]
    assert all(path.is_file() for path in paths.values())
    from PIL import Image

    assert all(Image.open(path).size == (512, 512) for path in paths.values())


def test_budget_exhaustion(tmp_path):
    backend = EmptyBackend()
    episode = run_episode(
        FakeEnv(),
        backend,
        tmp_path / "memory",
        Budgets(max_turns=1),
        tmp_path / "episode.jsonl",
    )
    assert episode["budget_exhausted"] == "turns"
    assert episode["turns"] == 1
    assert Path(tmp_path / "episode.jsonl").read_text()
    assert [message["role"] for message in backend.messages] == ["system", "user"]
    assert backend.messages[1]["content"] == "Begin the task."


def test_run_episode_uses_requested_system_prompt(tmp_path):
    prompt = tmp_path / "custom_prompt.txt"
    prompt.write_text("Use the supplied instructions.")
    backend = EmptyBackend()

    run_episode(
        FakeEnv(),
        backend,
        tmp_path / "memory",
        Budgets(max_turns=1),
        tmp_path / "episode.jsonl",
        system_prompt_path=prompt,
    )

    assert backend.messages[0]["content"] == "Use the supplied instructions."


def test_system_prompt_cli_is_threaded(monkeypatch, tmp_path):
    prompt = tmp_path / "custom_prompt.txt"
    output = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--tasks",
            "U0008",
            "--backend",
            "scripted",
            "--out",
            str(output),
            "--system-prompt",
            str(prompt),
            "--record-trajectory",
            "--trajectory-stride",
            "2",
            "--request-timeout",
            "42",
        ],
    )
    monkeypatch.setattr(
        harness_run, "resolve_attempts", lambda _target_ids: {"U0008": tmp_path}
    )

    class StubEnv:
        def __init__(
            self,
            _attempt_dir,
            _device,
            record_trajectory=False,
            trajectory_stride=5,
            target_mode="positions",
        ):
            self.plan = {}
            self.execution_order = []
            self.active_part_count = 1
            self.record_trajectory = record_trajectory
            self.trajectory_stride = trajectory_stride
            self.target_mode = target_mode

        def trajectory(self):
            return {
                "qpos": np.zeros((2, 3), dtype=np.float32),
                "qvel": np.zeros((2, 2), dtype=np.float32),
                "segments": [
                    {
                        "tool": "move_part",
                        "part": 0,
                        "args": {"part": 0},
                        "ok": True,
                        "reason": None,
                        "start": 0,
                        "end": 1,
                    }
                ],
            }

    captured = {}

    def fake_make_backend(*args):
        captured["request_timeout"] = args[-1]
        return object()

    monkeypatch.setattr(harness_run, "make_backend", fake_make_backend)

    def fake_env(attempt_dir, device, **kwargs):
        env = StubEnv(attempt_dir, device, **kwargs)
        captured["record_trajectory"] = env.record_trajectory
        captured["trajectory_stride"] = env.trajectory_stride
        return env

    def fake_run_episode(*args, **kwargs):
        captured["system_prompt_path"] = kwargs["system_prompt_path"]
        return {"passed": True, "parts_correct": 1}

    monkeypatch.setattr(harness_run, "CastleEnv", fake_env)
    monkeypatch.setattr(harness_run, "run_episode", fake_run_episode)

    harness_run.main()

    assert captured["system_prompt_path"] == prompt
    assert captured["request_timeout"] == 42
    assert captured["record_trajectory"] is True
    assert captured["trajectory_stride"] == 2
    trajectory_path = output / "episodes/000_U0008.trajectory.npz"
    with np.load(trajectory_path, allow_pickle=False) as saved:
        assert saved["qpos"].shape == (2, 3)
        assert saved["qvel"].shape == (2, 2)
        assert saved["attempt_dir"].item() == str(tmp_path)
        assert saved["stride"].item() == 2
        assert '"move_part"' in saved["segments"].item()


def test_resume_skips_completed_tasks_and_preserves_keep_memory(monkeypatch, tmp_path):
    output = tmp_path / "out"
    memory_dir = output / "memory"
    memory_dir.mkdir(parents=True)
    (memory_dir / "notes.md").write_text("keep this memory")
    episodes_dir = output / "episodes"
    episodes_dir.mkdir()
    (episodes_dir / "002_U0149.jsonl").write_text("partial turn\n")
    results_path = output / "results.jsonl"
    results_path.write_text(
        json.dumps({"target_id": "U0008"})
        + "\n"
        + json.dumps({"target_id": "U0080"})
        + "\n"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--tasks",
            "U0008,U0149,U0080",
            "--backend",
            "scripted",
            "--out",
            str(output),
            "--resume",
        ],
    )
    monkeypatch.setattr(
        harness_run,
        "resolve_attempts",
        lambda _target_ids: {
            target_id: tmp_path / target_id for target_id in ("U0008", "U0080", "U0149")
        },
    )
    monkeypatch.setattr(harness_run, "make_backend", lambda *args: object())
    captured = {"tasks": []}

    class StubEnv:
        def __init__(self, attempt_dir, _device, **_kwargs):
            self.target_id = Path(attempt_dir).name
            self.plan = {}
            self.execution_order = []
            self.active_part_count = 1

    def fake_run_episode(env, _backend, memory, _budgets, log_path, **_kwargs):
        captured["tasks"].append(env.target_id)
        captured["memory"] = (Path(memory) / "notes.md").read_text()
        captured["episode_path"] = Path(log_path)
        Path(log_path).write_text("resumed turn\n")
        return {"passed": True, "parts_correct": 1}

    monkeypatch.setattr(harness_run, "CastleEnv", StubEnv)
    monkeypatch.setattr(harness_run, "run_episode", fake_run_episode)

    harness_run.main()

    assert captured["tasks"] == ["U0149"]
    assert captured["memory"] == "keep this memory"
    assert captured["episode_path"] == episodes_dir / "002_U0149.jsonl"
    assert captured["episode_path"].read_text() == "resumed turn\n"
    assert (episodes_dir / "002_U0149.jsonl.partial").read_text() == "partial turn\n"
    rows = [json.loads(line) for line in results_path.read_text().splitlines()]
    assert [row["target_id"] for row in rows] == ["U0008", "U0080", "U0149"]
    assert rows[-1]["order_index"] == 2


def test_target_panel_is_composited_to_the_right():
    frame = np.zeros((384, 1024, 3), dtype=np.uint8)
    target = np.full((384, 384, 3), 127, dtype=np.uint8)

    result = _add_target_panel(frame, target)

    assert result.shape == (384, 1408, 3)
    np.testing.assert_array_equal(result[40, 1120], [127, 127, 127])
    np.testing.assert_array_equal(result[100, 100], [0, 0, 0])
    np.testing.assert_array_equal(result[0, 1034], [255, 255, 255])


def test_target_panel_preserves_native_high_resolution_size():
    frame = np.zeros((960, 1280, 3), dtype=np.uint8)
    target = np.full((960, 960, 3), 127, dtype=np.uint8)

    result = _add_target_panel(frame, target)

    assert result.shape == (960, 2240, 3)
    np.testing.assert_array_equal(result[400, 1280 + 400], [127, 127, 127])


def test_view_index_selects_one_camera_and_default_keeps_both():
    views = np.stack(
        (
            np.zeros((4, 5, 3), dtype=np.uint8),
            np.full((4, 5, 3), 127, dtype=np.uint8),
        )
    )

    np.testing.assert_array_equal(_select_view(views, 1), views[1])
    assert _select_view(views, None).shape == (4, 10, 3)
    with pytest.raises(ValueError, match="view-index must be between"):
        _select_view(views, 2)


def test_status_caption_scales_for_high_resolution_view():
    frame = np.zeros((960, 1280, 3), dtype=np.uint8)
    segments = [
        {
            "tool": "move_part",
            "part": 6,
            "ok": False,
            "reason": "recovery_state_not_safe",
            "start": 0,
            "end": 10,
        }
    ]

    result = _segment_caption(frame, 5, segments, font_scale=2.5)

    assert result.shape == frame.shape
    assert np.any(np.all(result[:100, :1000] == 255, axis=2))


def test_episode_text_maps_assistant_and_memory_text_to_segments():
    turns = [
        {
            "turn": 1,
            "messages": [{"role": "assistant", "content": "Inspecting the task."}],
            "tool_calls": [{"name": "observe", "arguments": {}}],
        },
        {
            "turn": 2,
            "messages": [{"role": "assistant", "content": "Saving a note."}],
            "tool_calls": [
                {
                    "name": "write_memory",
                    "arguments": {"path": "notes.md", "content": "abc"},
                },
                {
                    "name": "move_part",
                    "arguments": {
                        "part": 4,
                        "x": 0.33,
                        "y": -0.07,
                        "z": 0.02,
                    },
                },
            ],
        },
    ]
    segments = [
        {
            "tool": "move_part",
            "part": 4,
            "args": {
                "part": 4,
                "x": 0.33,
                "y": -0.07,
                "z": 0.02,
                "yaw_deg": 0.0,
            },
            "ok": True,
            "reason": None,
            "start": 0,
            "end": 2,
        }
    ]

    mapped = _map_episode_text_to_segments(turns, segments)

    assert mapped[0]["turn"] == 2
    assert mapped[0]["text"] == [
        "t1: Inspecting the task.",
        "t2: Saving a note.",
        "write_memory notes.md (+3 chars)",
    ]


def test_episode_text_mapping_raises_on_tool_argument_mismatch():
    turns = [
        {
            "turn": 3,
            "messages": [{"role": "assistant", "content": "Move it."}],
            "tool_calls": [
                {
                    "name": "move_part",
                    "arguments": {
                        "part": 4,
                        "x": 0.33,
                        "y": -0.07,
                        "z": 0.02,
                    },
                }
            ],
        }
    ]
    segments = [
        {
            "tool": "move_part",
            "part": 5,
            "args": {
                "part": 5,
                "x": 0.33,
                "y": -0.07,
                "z": 0.02,
                "yaw_deg": 0.0,
            },
            "ok": True,
            "reason": None,
            "start": 0,
            "end": 2,
        }
    ]

    with pytest.raises(ValueError, match="Episode log mismatch at turn 3"):
        _map_episode_text_to_segments(turns, segments)


def test_text_panel_is_composited_to_the_left_of_views_and_target():
    frame = np.zeros((384, 1024, 3), dtype=np.uint8)
    target = np.full((384, 384, 3), 64, dtype=np.uint8)
    frame = _add_target_panel(frame, target)
    panel = np.full((384, 384, 3), 127, dtype=np.uint8)

    result = _add_text_panel(frame, panel)

    assert result.shape == (384, 1792, 3)
    np.testing.assert_array_equal(result[40, 100], [127, 127, 127])
    np.testing.assert_array_equal(result[40, 500], [0, 0, 0])
    np.testing.assert_array_equal(result[40, 1500], [64, 64, 64])


def test_empty_text_uses_the_most_recent_text_in_gray():
    text, color, previous = _panel_text_content(
        {"text": ["t4: Placing the base."]}, None
    )
    assert text == ["t4: Placing the base."]
    assert color == (255, 255, 255)
    assert previous == "t4: Placing the base."

    text, color, previous = _panel_text_content({"text": []}, previous)
    assert text == ["(earlier) t4: Placing the base."]
    assert color == (128, 128, 128)
    assert previous == "t4: Placing the base."

    text, color, previous = _panel_text_content({"text": []}, None)
    assert text == ["(no comment)"]
    assert color == (255, 255, 255)
    assert previous is None


@pytest.mark.parametrize(
    ("tasks", "expected"),
    [
        (None, "U0008,U0080,U0149,U0002,U0007,U0011,U0012,U0024,U0052"),
        ("U0008", "U0008"),
        ("S0074,U0008", "S0074,U0008"),
    ],
)
def test_runner_task_selection(monkeypatch, tasks, expected):
    arguments = ["run.py", "--backend", "scripted"]
    if tasks is not None:
        arguments.extend(["--tasks", tasks])
    monkeypatch.setattr(sys, "argv", arguments)
    assert harness_run.parse_args().tasks == expected


def test_trajectory_recording_is_disabled_by_default(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "--tasks", "U0008", "--backend", "scripted"],
    )
    args = harness_run.parse_args()
    assert args.record_trajectory is False
    assert args.trajectory_stride == 5
    assert args.resume is False
    assert args.request_timeout == 600
    assert args.target_mode == "positions"
    assert (
        inspect.signature(harness_run.CastleEnv).parameters["record_trajectory"].default
        is False
    )
    assert harness_run.resolve_system_prompt("positions", None) == (
        Path(harness_run.__file__).resolve().parent / "prompts/system_level1.txt"
    )
    assert harness_run.resolve_system_prompt("image", None) == (
        Path(harness_run.__file__).resolve().parent / "prompts/system_image.txt"
    )


def test_trajectory_buffer_samples_states_and_tracks_segments():
    buffer = _TrajectoryBuffer(stride=2)
    initial_qpos = np.array([0, 1, 2], dtype=np.float64)
    initial_qvel = np.array([0, 1], dtype=np.float64)
    buffer.append(initial_qpos, initial_qvel)
    start = buffer.last_index
    stage_qpos = np.arange(21, dtype=np.float64).reshape(7, 3)
    stage_qvel = np.arange(14, dtype=np.float64).reshape(7, 2)
    buffer.append_episode(stage_qpos, stage_qvel)
    failed_move = {"ok": False, "reason": "planning_cpu_budget"}
    buffer.finish_segment(
        start,
        "move_part",
        0,
        {"part": 0},
        failed_move,
        stage_qpos[-1],
        stage_qvel[-1],
    )
    next_start = buffer.last_index
    buffer.finish_segment(
        next_start,
        "step",
        None,
        {"action": [0.0] * 7},
        {"ok": True},
        stage_qpos[-1],
        stage_qvel[-1],
    )

    trajectory = buffer.as_dict()
    assert trajectory["qpos"].dtype == np.float32
    assert trajectory["qvel"].dtype == np.float32
    np.testing.assert_array_equal(
        trajectory["qpos"],
        np.vstack((initial_qpos, stage_qpos[[0, 2, 4, 6]], stage_qpos[-1])),
    )
    segments = trajectory["segments"]
    assert segments[0]["reason"] == "planning_cpu_budget"
    assert segments[0]["start"] < segments[0]["end"]
    assert segments[1]["start"] >= segments[0]["end"]
    assert segments[1]["start"] < segments[1]["end"]


def test_result_row_and_failure_categories():
    base = {"passed": False, "parts_correct": 6, "sim_steps": 70, "wall_s": 1}
    assert failure_category({**base, "budget_exhausted": "turns"}) == (
        "budget_exhausted:turns"
    )
    assert failure_category({**base, "fatal": True}) == "sim_fatal"
    assert failure_category({**base, "tool_error": True}) == "tool_error"
    assert failure_category(
        {**base, "hold": {"pose_valid": True, "settled": False}}
    ) == ("unstable")
    assert (
        failure_category({**base, "per_part_errors": [{"upright": False}]})
        == "knocked_over"
    )
    assert (
        failure_category({**base, "failing_checks": ["checker_error:ExecutionGuard"]})
        == "tool_error"
    )
    assert (
        failure_category(
            {
                **base,
                "failing_checks": ["checker_error:ExecutionGuard"],
                "per_part_errors": [{"upright": False}],
            }
        )
        == "knocked_over"
    )
    assert failure_category(base) == "wrong_pose"
    row = result_row("U1", 7, 0, "wipe", "scripted", "", base)
    assert row["parts_correct"] == 6
    assert row["sim_steps_per_part"] == 10


def test_scripted_backend_uses_recorded_execution_order():
    backend = ScriptedBackend(
        {
            "assembly_order": [0, 1],
            "target": [[0, 0, 0], [1, 1, 1]],
        },
        execution_order=[1, 0],
    )
    first, _ = backend.respond([], [])
    second, _ = backend.respond([], [])
    final, _ = backend.respond([], [])
    assert first["tool_calls"][0]["name"] == "get_task"
    assert second["tool_calls"][0]["arguments"]["part"] == 1
    assert final["tool_calls"][0]["arguments"]["part"] == 0


def test_anthropic_fake_http(monkeypatch):
    captured = {}

    def handler(url, headers, payload):
        captured.update(payload)
        captured["url"] = url
        captured["headers"] = headers
        return {
            "content": [
                {
                    "type": "tool_use",
                    "id": "a1",
                    "name": "observe",
                    "input": {},
                }
            ],
            "usage": {"input_tokens": 11, "output_tokens": 4},
        }

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    backend = AnthropicBackend(
        "claude-test", base_url="https://fake", requester=handler
    )
    turn, usage = backend.respond(
        [{"role": "system", "content": "system"}, {"role": "user", "content": "go"}],
        [],
    )
    assert captured["system"] == "system"
    assert turn["tool_calls"][0]["name"] == "observe"
    assert usage == {"input_tokens": 11, "output_tokens": 4}


def test_openai_fake_http():
    captured = {}

    def handler(url, headers, payload):
        captured.update(payload)
        captured["url"] = url
        captured["headers"] = headers
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "o1",
                                "type": "function",
                                "function": {
                                    "name": "move_part",
                                    "arguments": (
                                        '{"part": 2, "x": 1, "y": 0, "z": 0}'
                                    ),
                                },
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 3},
        }

    backend = OpenAIBackend("qwen", "http://fake/v1", requester=handler)
    turn, usage = backend.respond([{"role": "system", "content": "system"}], [])
    assert captured["model"] == "qwen"
    assert turn["tool_calls"][0]["arguments"]["part"] == 2
    assert usage == {"input_tokens": 8, "output_tokens": 3}


def test_openai_timeout_retries_and_logs_event(monkeypatch, tmp_path):
    calls = []
    backoffs = []

    response_payload = {
        "choices": [{"message": {"content": "thinking", "tool_calls": []}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    }

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps(response_payload).encode()

    def fake_urlopen(_call, timeout):
        calls.append(timeout)
        if len(calls) == 1:
            raise TimeoutError("request timed out")
        return FakeResponse()

    monkeypatch.setattr(harness_backends.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(harness_backends.time, "sleep", backoffs.append)
    backend = OpenAIBackend(
        "qwen",
        "http://fake/v1",
        request_timeout=42,
    )

    episode_path = tmp_path / "episode.jsonl"
    run_episode(
        FakeEnv(),
        backend,
        tmp_path / "memory",
        Budgets(max_turns=1),
        episode_path,
    )

    records = [json.loads(line) for line in episode_path.read_text().splitlines()]
    assert calls == [42, 42]
    assert backoffs == [10]
    assert records[0] == {
        "event": "http_retry",
        "attempt": 1,
        "error": "TimeoutError: request timed out",
    }
    assert records[1]["turn"] == 1


def test_openai_http_400_is_not_retried(monkeypatch):
    calls = []
    backoffs = []
    events = []

    def fake_urlopen(call, timeout):
        calls.append(timeout)
        raise HTTPError(call.full_url, 400, "bad request", None, None)

    monkeypatch.setattr(harness_backends.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(harness_backends.time, "sleep", backoffs.append)

    with pytest.raises(HTTPError) as raised:
        harness_backends._post_json(
            "http://fake/v1",
            {},
            {},
            timeout=42,
            on_retry=events.append,
        )

    assert raised.value.code == 400
    assert calls == [42]
    assert backoffs == []
    assert events == []


class TwoMovesBackend:
    def __init__(self):
        self.tools = []

    def respond(self, _messages, tools):
        self.tools.append(tools)
        if len(self.tools) == 1:
            calls = [
                {"id": f"move-{part}", "name": "move_part",
                 "arguments": {"part": part, "x": 0, "y": 0, "z": 0}}
                for part in (0, 1)
            ]
        else:
            calls = [{"id": "submit", "name": "submit", "arguments": {}}]
        return {"content": "", "tool_calls": calls}, {
            "input_tokens": 1,
            "output_tokens": 1,
        }


def test_only_one_move_part_runs_per_turn(tmp_path):
    env = FakeEnv()
    run_episode(
        env,
        TwoMovesBackend(),
        tmp_path / "memory",
        Budgets(max_turns=3),
        tmp_path / "episode.jsonl",
    )
    assert env.sim_steps == 3
    first = json.loads((tmp_path / "episode.jsonl").read_text().splitlines()[0])
    assert first["tool_results"][0]["result"]["ok"]
    assert first["tool_results"][1]["result"] == {
        "ok": False,
        "error": ONE_MOVE_PER_TURN_ERROR,
    }


def test_episode_log_records_only_new_messages(tmp_path):
    run_episode(
        FakeEnv(),
        EmptyBackend(),
        tmp_path / "memory",
        Budgets(max_turns=2),
        tmp_path / "episode.jsonl",
    )
    records = [
        json.loads(line)
        for line in (tmp_path / "episode.jsonl").read_text().splitlines()
    ]
    assert [m["role"] for m in records[0]["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    assert [m["role"] for m in records[1]["messages"]] == ["assistant"]


def test_image_mode_get_task_description_omits_positions(tmp_path):
    env = FakeEnv()
    env.target_mode = "image"
    backend = TwoMovesBackend()
    run_episode(
        env, backend, tmp_path / "memory", Budgets(max_turns=1), tmp_path / "e.jsonl"
    )
    get_task = next(
        tool for tool in backend.tools[0] if tool["function"]["name"] == "get_task"
    )
    assert "not given" in get_task["function"]["description"]


def test_run_python_does_not_inherit_harness_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("CASTLE_ROOT", "/private/castle")
    dispatcher = ToolDispatcher(FakeEnv(), tmp_path / "memory", tmp_path / "scratch")
    output = dispatcher.dispatch(
        "run_python", {"code": "import os; print(os.environ.get('CASTLE_ROOT'))"}
    )
    assert output["stdout"].strip() == "None"


def test_episode_scratch_directory_is_removed(tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"

    def fake_mkdtemp(prefix):
        scratch.mkdir()
        return str(scratch)

    monkeypatch.setattr(harness_agent.tempfile, "mkdtemp", fake_mkdtemp)
    run_episode(
        FakeEnv(),
        EmptyBackend(),
        tmp_path / "memory",
        Budgets(max_turns=1),
        tmp_path / "episode.jsonl",
    )
    assert not scratch.exists()
    assert (tmp_path / "memory").exists()
