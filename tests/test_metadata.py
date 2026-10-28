"""The package metadata: Python classifiers agree with requires-python and with the versions CI tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CI = REPO / ".github" / "workflows" / "ci.yml"


def test_python_classifiers() -> None:
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    minimum = re.search(r'requires-python = ">=3\.(\d+)"', pyproject)
    assert minimum is not None
    listed = sorted(int(v) for v in re.findall(r'"Programming Language :: Python :: 3\.(\d+)"', pyproject))
    assert listed == list(range(int(minimum.group(1)), listed[-1] + 1))  # from the minimum, without holes
    assert "License ::" not in pyproject  # the licence is the SPDX expression; setuptools refuses both
    if not CI.is_file():
        pytest.skip("no .github folder (e.g. tests from the sdist)")
    tested = {int(v) for v in re.findall(r'"3\.(\d+)"', CI.read_text(encoding="utf-8"))}
    assert tested <= set(listed)
