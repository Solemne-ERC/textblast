"""Build helpers (no compiler needed) and the CLI surface."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from textblast import cli, logs
from textblast.engine.build import (
    EXCLUDED_PROJECTS,
    BuildError,
    BuildOptions,
    _project_list,
    _unmet_requirements,
    fingerprint,
    resolve_openmp,
)
from textblast.engine.versions import RELEASES, get_release


def test_fingerprint_depends_on_build_choices() -> None:
    release = get_release("2.17.0")
    base = fingerprint(release, BuildOptions("off", "restricted"))
    assert base == fingerprint(release, BuildOptions("off", "restricted"))
    assert base != fingerprint(release, BuildOptions("on", "restricted"))
    assert base != fingerprint(release, BuildOptions("off", "full"))
    assert base != fingerprint(get_release("2.13.0"), BuildOptions("off", "restricted"))


def test_openmp_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.platform", "darwin")
    assert resolve_openmp("auto") == "off"
    monkeypatch.setattr("sys.platform", "linux")
    assert resolve_openmp("auto") == "on"
    assert resolve_openmp("off") == "off"
    with pytest.raises(ValueError):
        resolve_openmp("maybe")


def test_configure_args() -> None:
    assert BuildOptions("off", "full").configure_args(None) == ["--without-openmp"]
    restricted = BuildOptions("on", "restricted").configure_args(Path("/x.lst"))
    assert restricted == ["--with-openmp", "--with-projects=/x.lst"]


def test_versions_table() -> None:
    assert RELEASES["2.17.0"].role == "default" and "bitmagic_allocator" not in RELEASES["2.17.0"].patches
    assert "bitmagic_allocator" in RELEASES["2.13.0"].patches
    assert RELEASES["2.17.0"].url.endswith("/2.17.0/ncbi-blast-2.17.0+-src.tar.gz")
    with pytest.raises(ValueError, match="unsupported"):
        get_release("1.0.0")


def test_restricted_project_list(tmp_path: Path) -> None:
    lst = tmp_path / "scripts/projects/blast/project.lst"
    lst.parent.mkdir(parents=True)
    lst.write_text("corelib$\nalgo/blast/api\nalgo/blast/unit_tests\napp/blast\napp/dustmasker$\n-objmgr/test\n")
    out = _project_list(tmp_path, tmp_path).read_text().splitlines()
    assert "algo/blast/unit_tests" not in out and "app/dustmasker$" not in out
    assert {"corelib$", "algo/blast/api", "app/blast", "-objmgr/test"} <= set(out)
    assert {f"-{p}" for p in EXCLUDED_PROJECTS} <= set(out)


def test_build_error_log_tail(tmp_path: Path) -> None:
    log = tmp_path / "build.log"
    log.write_text("".join(f"line {i}\n" for i in range(100)))
    tail = BuildError("make failed", log).log_tail()
    assert len(tail) == 40 and tail[-1] == "line 99"


def test_unmet_requirements_name_the_missing_library(tmp_path: Path) -> None:
    log = tmp_path / "build.log"
    line = "make[4]: *** No rule to make target '/b/ReleaseMT/status/{}.enabled', needed by 'requirements'.  Stop.\n"
    log.write_text(line.format("SQLITE3") * 3 + line.format("LIBFOO") + "NOTE:  skipping project \"blastp\"\n")
    message = _unmet_requirements(log)
    assert "SQLITE3 (sqlite-devel or libsqlite3-dev)" in message and "LIBFOO" in message
    log.write_text("all fine\n")
    assert _unmet_requirements(log) == "" and _unmet_requirements(tmp_path / "missing.log") == ""


def test_cli_help_and_version(capsys: pytest.CaptureFixture[str]) -> None:
    for argv in (["--version"], ["engine", "build", "--help"], ["engine", "check", "--help"]):
        with pytest.raises(SystemExit) as exit_info:
            cli.main(argv)
        assert exit_info.value.code == 0
    assert "0.1.0" in capsys.readouterr().out


def test_cli_reports_missing_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEXTBLAST_BIN", raising=False)
    stream = io.StringIO()
    monkeypatch.setattr(cli, "setup_logging", lambda verbosity: logs.setup_logging(verbosity, stream=stream))
    code = cli.main(["engine", "info", "--cache", str(tmp_path)])
    assert code == 1
    assert "error: no BLAST 2.17.0 engine found" in stream.getvalue()
