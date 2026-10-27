"""Engine lookup: argument, environment, bundled engine, cache; a bundled engine's check results; no engine at all."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

from textblast import logs
from textblast.config import EngineOptions, RunConfig
from textblast.engine import build, check, locate
from textblast.engine.locate import EngineError, bundled_engine_dir, find_engine
from textblast.engine.versions import BINARIES
from textblast.plan import _engine_record
from textblast.run import run_all


def fake_engine(root: Path, fingerprint: str, version: str = "2.17.0", managed: bool = True) -> Path:
    """An engine directory whose binaries only answer ``-version``."""
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True)
    for name in BINARIES:
        script = bin_dir / name
        script.write_text(f"#!/bin/sh\necho '{name}: {version}+'\necho ' Package: blast {version}, build fake'\n")
        script.chmod(0o755)
    if managed:
        info = {"version": version, "fingerprint": fingerprint, "built_at": "2026-01-01T00:00:00+00:00"}
        (root / "engine.json").write_text(json.dumps(info))
    return root


@pytest.fixture
def engines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """One fake engine per source, with the bundle reported by the package and the cache in tmp_path."""
    paths = {
        "argument": fake_engine(tmp_path / "arg", "fp-argument"),
        "environment": fake_engine(tmp_path / "env", "fp-environment"),
        "bundled": fake_engine(tmp_path / "site" / "textblast" / "_engine", "fp-bundled"),
        "cache": fake_engine(tmp_path / "cache" / "engine" / "2.17.0-fp-cache", "fp-cache"),
    }
    monkeypatch.setenv("TEXTBLAST_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("TEXTBLAST_BIN", raising=False)
    monkeypatch.setattr(locate, "bundled_engine_dir", lambda package="textblast": paths["bundled"])
    logs.setup_logging(stream=io.StringIO())
    return paths


def test_lookup_order(engines: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEXTBLAST_BIN", str(engines["environment"] / "bin"))
    found = find_engine(engines["argument"] / "bin")
    assert (found.source, found.fingerprint) == ("argument", "fp-argument")
    found = find_engine()
    assert (found.source, found.fingerprint) == ("environment", "fp-environment")
    monkeypatch.delenv("TEXTBLAST_BIN")
    found = find_engine()
    assert (found.source, found.fingerprint) == ("bundled", "fp-bundled")
    assert _engine_record(found)["source"] == "bundled"  # what the plan and the manifest record
    monkeypatch.setattr(locate, "bundled_engine_dir", lambda package="textblast": None)
    found = find_engine()
    assert (found.source, found.fingerprint) == ("cache", "fp-cache")
    assert _engine_record(found)["source"] == "cache"


def test_bundle_of_another_version_is_skipped(engines: dict[str, Path], tmp_path: Path) -> None:
    fake_engine(tmp_path / "cache" / "engine" / "2.13.0-fp-ref", "fp-ref", version="2.13.0")
    found = find_engine(version="2.13.0")
    assert (found.source, found.fingerprint) == ("cache", "fp-ref")


def test_broken_bundle_falls_through_to_the_cache(engines: dict[str, Path]) -> None:
    stream = io.StringIO()
    logs.setup_logging(stream=stream)
    (engines["bundled"] / "bin" / "blastp").unlink()
    found = find_engine()
    assert found.source == "cache"
    assert "bundled engine" in stream.getvalue() and "unusable" in stream.getvalue()
    assert "missing binaries" in stream.getvalue() and "looking in the cache" in stream.getvalue()


def test_bundled_check_results_go_to_the_cache(engines: dict[str, Path], tmp_path: Path) -> None:
    bundle_info = (engines["bundled"] / "engine.json").read_text()
    engine = find_engine()
    assert engine.info.get("last_check") is None
    report = check.CheckReport(engine={}, results=[], checked_at="2026-01-01T00:00:00+00:00")
    check._record(engine, report)
    assert (engines["bundled"] / "engine.json").read_text() == bundle_info  # the installed package is untouched
    assert engine.check_path is not None and engine.check_path.is_relative_to(tmp_path / "cache")
    assert find_engine().info["last_check"]["passed"] is True
    # A record left by another build at the same place (an older wheel) does not count.
    record = json.loads(engine.check_path.read_text())
    engine.check_path.write_text(json.dumps({**record, "fingerprint": "fp-older"}))
    assert find_engine().info["last_check"] is None
    # The same bundle reached through --blast-bin keeps its check results in the cache too.
    assert find_engine(engines["bundled"] / "bin").check_path == engine.check_path


def test_bundled_engine_dir_in_an_installed_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    site = tmp_path / "site"
    (site / "withbundle").mkdir(parents=True)
    (site / "withbundle" / "__init__.py").write_text("")
    fake_engine(site / "withbundle" / "_engine", "fp")
    (site / "nobundle").mkdir()
    (site / "nobundle" / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(site))
    try:
        assert bundled_engine_dir("withbundle") == site / "withbundle" / "_engine"
        assert bundled_engine_dir("nobundle") is None
        assert bundled_engine_dir("not_installed_anywhere") is None
    finally:
        for name in ("withbundle", "nobundle"):
            sys.modules.pop(name, None)


def test_no_engine_says_to_build_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An install from source (no bundle) with an empty cache."""
    monkeypatch.delenv("TEXTBLAST_BIN", raising=False)
    monkeypatch.setenv("TEXTBLAST_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(locate, "bundled_engine_dir", lambda package="textblast": None)
    with pytest.raises(EngineError, match=r"none bundled .* run `textblast engine build` once"):
        find_engine()
    with pytest.raises(EngineError, match=r"run `textblast engine build --version 2\.13\.0` once"):
        find_engine(version="2.13.0")
    # `run` says so before preparing the corpus.
    config = RunConfig(input=str(tmp_path / "missing.jsonl"), run_dir=str(tmp_path / "run"), engine=EngineOptions())
    with pytest.raises(EngineError, match="textblast engine build"):
        run_all(config)
    assert not (tmp_path / "run").exists()


def test_build_notes_a_bundle_of_the_same_version(
    engines: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(build, "bundled_engine_dir", lambda package="textblast": engines["bundled"])
    stream = io.StringIO()
    logs.setup_logging(stream=stream)
    build._note_bundle("2.17.0", tmp_path / "built")
    assert f"--blast-bin {tmp_path / 'built' / 'bin'}" in stream.getvalue()
    stream.truncate(0)
    build._note_bundle("2.13.0", tmp_path / "built")
    assert stream.getvalue() == ""
