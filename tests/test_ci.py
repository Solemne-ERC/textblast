"""The CI workflow's engine cache: its key is the engine fingerprint, and every job uses the same key and paths."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

from textblast.engine.build import BuildOptions, fingerprint, resolve_openmp
from textblast.engine.versions import DEFAULT_VERSION, get_release

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"
KEY_SCRIPT = REPO / ".github" / "scripts" / "engine_key.py"


def test_engine_key_is_the_fingerprint(tmp_path: Path) -> None:
    output = tmp_path / "github_output"
    env = {**os.environ, "GITHUB_OUTPUT": str(output), "TEXTBLAST_CACHE": str(tmp_path / "cache")}
    subprocess.run([sys.executable, str(KEY_SCRIPT)], check=True, capture_output=True, env=env)
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    expected = fingerprint(get_release(DEFAULT_VERSION), BuildOptions(resolve_openmp("auto"), "restricted"))
    assert values["fingerprint"] == expected
    assert values["key"] == f"engine-{DEFAULT_VERSION}-{expected}"
    assert values["engine_dir"] == str(tmp_path / "cache" / "engine" / f"{DEFAULT_VERSION}-{expected}")
    assert values["tarball"] == str(tmp_path / "cache" / "src" / get_release(DEFAULT_VERSION).tarball_name)


def test_workflow_jobs_share_the_engine_cache() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    caches = [
        step["with"]
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if step.get("uses", "").startswith("actions/cache/")
    ]
    assert len(caches) == 3  # restore and save in `engine`, restore in `tests`
    assert len({(c["key"], c["path"]) for c in caches}) == 1
    assert workflow["jobs"]["tests"]["needs"] == "engine"
    assert caches[-1]["fail-on-cache-miss"] is True
