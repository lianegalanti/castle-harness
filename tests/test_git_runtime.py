"""The installed runtime comes from the reviewed public Git revision."""

import importlib.metadata
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_installed_packages_match_the_public_git_pins():
    manifest = tomllib.loads((ROOT / "pixi.toml").read_text())
    packages = manifest["pypi-dependencies"]
    benchmark = packages["castle-reasoning-benchmark"]
    runtime = packages["mujoco-warp"]
    assert benchmark["git"] == runtime["git"]
    assert benchmark["rev"] == runtime["rev"]
    assert len(benchmark["rev"]) == 40
    assert runtime["subdirectory"] == "third_party/mujoco-warp"
    for name in ("castle-reasoning-benchmark", "mujoco-warp"):
        specification = packages[name]
        installed = importlib.metadata.distribution(name)
        origin = json.loads(installed.read_text("direct_url.json"))
        assert origin["url"] == specification["git"]
        assert origin["vcs_info"]["vcs"] == "git"
        assert origin["vcs_info"]["commit_id"] == specification["rev"]
        assert origin.get("subdirectory") == specification.get("subdirectory")
        assert not origin.get("dir_info", {}).get("editable", False)
