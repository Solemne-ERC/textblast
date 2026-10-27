"""The package was renamed from solemne-blast to textblast before 0.9.0: no old name is left in code or configs."""

from __future__ import annotations

from pathlib import Path

import pytest

from textblast.config import BlastOptions, resolved_settings

ROOT = Path(__file__).resolve().parents[1]
OLD = ("solemne_blast", "solemne-blast", "SOLEMNE_BLAST")


def test_no_old_name_in_code_or_configs() -> None:
    files = [*(ROOT / "src" / "textblast").rglob("*.py"), *(ROOT / "configs").glob("*.yaml"), ROOT / "pyproject.toml"]
    found = [f"{path.relative_to(ROOT)}:{n}" for path in files
             for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
             if any(old in line for old in OLD) and "RENAMED_PRESETS" not in line]
    assert not found, f"old name in {found}"


def test_old_preset_name_is_refused_with_its_new_name() -> None:
    with pytest.raises(ValueError, match="now called 'textblast'"):
        resolved_settings(BlastOptions(preset="solemne-blast"))
