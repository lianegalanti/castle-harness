"""Harness commands use installed code and a separate downloaded data root."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER_TASKS = "U0008,U0080,U0149,U0002,U0007,U0011,U0012,U0024,U0052"


def test_runner_defaults_to_dataset_without_creating_it():
    existed = (ROOT / "dataset").exists()
    environment = dict(os.environ)
    environment.pop("CASTLE_ROOT", None)
    environment.pop("CASTLE_INDEX", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, run; print(json.dumps([str(p) for p in run.INDEXES]))",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [
        str(ROOT / "dataset/campaign-1k-20261010/index.json"),
        str(ROOT / "dataset/campaign-tasks-20261011/index.json"),
    ]
    assert (ROOT / "dataset").exists() == existed


@pytest.mark.parametrize("tasks", [None, "", "U0008"])
def test_scripted_runner_uses_environment_python(tmp_path, tasks):
    binary = tmp_path / "bin"
    binary.mkdir()
    receipt = tmp_path / "argv.json"
    executable = binary / "python"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "if sys.argv[1:2] == ['-c']:\n"
        "    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
        "pathlib.Path(os.environ['ARGV_LOG']).write_text(json.dumps({"
        "'argv': sys.argv[1:], 'pythonpath': os.environ.get('PYTHONPATH')}))\n"
    )
    executable.chmod(0o755)
    environment = dict(
        os.environ,
        PATH=f"{binary}:{os.environ['PATH']}",
        ARGV_LOG=str(receipt),
        CASTLE_ROOT=str(tmp_path / "data"),
        OUT_ROOT=str(tmp_path / "runs"),
    )
    environment.pop("TASKS", None)
    if tasks is not None:
        environment["TASKS"] = tasks
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/run_scripted.sh")],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    recorded = json.loads(receipt.read_text())
    assert recorded["argv"][0] == str(ROOT / "run.py")
    assert recorded["argv"][recorded["argv"].index("--tasks") + 1] == (
        tasks or PAPER_TASKS
    )
    assert recorded["pythonpath"] == os.environ.get("PYTHONPATH")
    assert not (tmp_path / "data").exists()


def test_python_tool_uses_harness_interpreter(monkeypatch, tmp_path):
    from test_harness import FakeEnv

    from castle_harness.tools import ToolDispatcher

    dispatcher = ToolDispatcher(
        FakeEnv(), tmp_path / "memory", tmp_path / "scratch"
    )
    monkeypatch.setenv("PATH", "")
    result = dispatcher.dispatch("run_python", {"code": "print(6 * 7)"})
    assert result["returncode"] == 0
    assert result["stdout"].strip() == "42"


def test_harness_import_does_not_load_demonstration_teacher():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import castle_harness.env; "
            "assert not any(n.startswith('castle_reasoning.data.assembly') "
            "for n in sys.modules)",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_harness_requested_and_terminal_stages_use_runtime_coordinator(
    tmp_path,
):
    from types import SimpleNamespace
    import numpy as np
    from castle_harness.env import CastleEnv, _OneWorldGroup
    from castle_reasoning.runtime.execution import session

    snapshot = {"qpos": np.zeros((1, 28)), "qvel": np.zeros((1, 26))}
    resets, operations = [], []

    def reset(**state):
        resets.append(state)

    class OperationGroup(_OneWorldGroup):
        def run_world(self, world, name, goals, order):
            operations.append((name, goals.copy(), list(order)))
            self.stage.finish(world)
            count = 51 if not order else 30
            episode = {
                "qpos": np.zeros((count + 1, 28)),
                "qvel": np.zeros((count + 1, 26)),
                "action": np.zeros((count, 7)),
            }
            return {"passed": True, "recorded_control_steps": count}, episode

    engine = SimpleNamespace(
        native_models=[object()], nworld=1, snapshot=lambda: snapshot
    )
    collector = SimpleNamespace(engine=engine, controller=object(), reset=reset)
    group = OperationGroup(collector, float("inf"), [tmp_path / "model.mjb"])
    original = {"name": "original"}
    state = object.__new__(CastleEnv)
    state.group = group
    state.runner = SimpleNamespace(plan=original)
    state.group.runners = [state.runner]
    state.sim_steps = 0
    state._trajectory = None
    assert callable(session.Runner.stage)
    for name, order in (("move_0_0", [1, 0]), ("terminal_check", [])):
        goals = np.zeros((2, 3))
        check, episode, used = state._run_stage(
            name, goals, order, {"name": name}
        )
        assert check["passed"] and used == len(episode["action"])
        assert state.runner.plan is original
    assert [operation[2] for operation in operations] == [[1, 0], []]
    assert len(resets) == 2 and state.sim_steps == 81


def test_mixed_downloaded_campaigns_resolve_in_request_order(
    monkeypatch, tmp_path
):
    import run

    indexes = []
    for campaign, target in (
        ("campaign-1k-20261010", "S0074"),
        ("campaign-tasks-20261011", "U0008"),
    ):
        directory = tmp_path / campaign
        attempt = directory / "shards/same/run00/attempt00000"
        attempt.mkdir(parents=True)
        (attempt / "plan.json").write_text(json.dumps({"target_id": target}))
        index = directory / "index.json"
        index.write_text(
            json.dumps(
                {
                    "demonstrations": [
                        {
                            "target_id": target,
                            "attempt_directory": "shards/same/run00/attempt00000",
                        }
                    ]
                }
            )
        )
        indexes.append(index)
    monkeypatch.setattr(run, "INDEXES", indexes)
    resolved = run.resolve_attempts(["U0008", "S0074"])
    assert list(resolved) == ["U0008", "S0074"]
    for target, path in resolved.items():
        assert (
            json.loads((path / "plan.json").read_text())["target_id"] == target
        )


def test_single_downloaded_campaign_can_use_later_available_attempt(
    monkeypatch, tmp_path
):
    import pytest

    import run

    directory = tmp_path / "campaign-1k-20261010"
    attempt = directory / "second"
    attempt.mkdir(parents=True)
    (attempt / "plan.json").write_text("{}")
    index = directory / "index.json"
    index.write_text(
        json.dumps(
            {
                "demonstrations": [
                    {"target_id": "S0074", "attempt_directory": "first"},
                    {"target_id": "S0074", "attempt_directory": "second"},
                ]
            }
        )
    )
    monkeypatch.setattr(run, "INDEXES", [index, tmp_path / "absent/index.json"])
    assert run.resolve_attempts(["S0074"]) == {"S0074": attempt}
    with pytest.raises(FileNotFoundError, match="U0008"):
        run.resolve_attempts(["U0008"])


def test_explicit_index_overrides_both_packaged_defaults(tmp_path):
    custom = tmp_path / "custom/index.json"
    environment = dict(
        os.environ, CASTLE_ROOT=str(tmp_path / "data"), CASTLE_INDEX=str(custom)
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, run; print(json.dumps([str(p) for p in run.INDEXES]))",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [str(custom)]
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize(
    ("tasks", "arguments", "expected", "root"),
    [
        (None, [], PAPER_TASKS, None),
        ("", [], PAPER_TASKS, None),
        ("S0074,U0008", [], "S0074,U0008", None),
        ("U0008", ["--tasks", "S0074"], "S0074", None),
        (None, ["--root", "custom-data"], PAPER_TASKS, "custom-data"),
        (None, ["--tasks", "U0080", "--root", "custom-data"], "U0080", "custom-data"),
    ],
)
def test_download_script_calls_installed_castle_reasoning_cli(
    tmp_path, tasks, arguments, expected, root
):
    binary = tmp_path / "bin"
    binary.mkdir()
    receipt = tmp_path / "argv.json"
    executable = binary / "python"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "if sys.argv[1:2] == ['-c']:\n"
        "    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
        "recorded = {'argv': sys.argv[1:], 'pythonpath': os.environ.get('PYTHONPATH')}\n"
        "from castle_reasoning import cli\n"
        "def download(tasks, root, **kwargs):\n"
        "    recorded.update(tasks=tasks, root=root)\n"
        "    return {'repo_id': 'fixture', 'revision': 'fixture', 'files': {},\n"
        "            'tasks': dict.fromkeys(tasks), 'selected_bytes': 0}\n"
        "cli.data_download.download_tasks = download\n"
        "sys.argv = ['castle-benchmark', *sys.argv[3:]]\n"
        "cli.main()\n"
        "pathlib.Path(os.environ['ARGV_LOG']).write_text(json.dumps(recorded))\n"
    )
    executable.chmod(0o755)
    environment = dict(
        os.environ,
        PATH=f"{binary}:{os.environ['PATH']}",
        ARGV_LOG=str(receipt),
    )
    environment.pop("TASKS", None)
    if tasks is not None:
        environment["TASKS"] = tasks
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/download_tasks.sh"), *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    recorded = json.loads(receipt.read_text())
    assert recorded["argv"][:3] == [
        "-m",
        "castle_reasoning.cli",
        "download",
    ]
    assert recorded["tasks"] == expected.split(",")
    assert recorded["root"] == root
    assert recorded["pythonpath"] == os.environ.get("PYTHONPATH")
