"""Logging foundation: formatting, stage lines, progress (non-TTY), and records from worker processes."""

from __future__ import annotations

import io
import logging
from concurrent.futures import ProcessPoolExecutor

import pytest

from textblast import logs
from textblast.logs import (
    QUIET,
    LogListener,
    Progress,
    Stage,
    configure_worker_logging,
    format_bytes,
    format_count,
    format_duration,
    get_logger,
    setup_logging,
)


@pytest.fixture
def console() -> io.StringIO:
    stream = io.StringIO()
    setup_logging(stream=stream)
    yield stream
    for handler in list(logs.logger.handlers):
        if getattr(handler, "_textblast", False):
            logs.remove_handler(handler)
    logs._console_handler = None


@pytest.mark.parametrize(
    ("seconds", "text"), [(3.21, "3.2s"), (42, "42s"), (125, "2m05s"), (25 * 60, "25m"), (72 * 60, "1h12m")]
)
def test_format_duration(seconds: float, text: str) -> None:
    assert format_duration(seconds) == text


def test_format_bytes_and_count() -> None:
    assert format_bytes(512) == "512 B"
    assert format_bytes(3 * 1024 * 1024) == "3.0 MB"
    assert format_count(950) == "950"
    assert format_count(12_300) == "12.3k"
    assert format_count(4_200_000) == "4.2M"


def test_library_does_not_print_without_setup(capsys: pytest.CaptureFixture[str]) -> None:
    get_logger("x").info("silent")
    assert capsys.readouterr().err == ""


def test_stage_lines(console: io.StringIO) -> None:
    with Stage(2, 5, "make", "all_p") as stage:
        stage.summary = "3 binaries"
    lines = console.getvalue().splitlines()
    assert lines[0] == "[2/5] make: all_p"
    assert lines[1].startswith("[2/5] make done in ") and lines[1].endswith(": 3 binaries")


def test_stage_failure_line(console: io.StringIO) -> None:
    with pytest.raises(RuntimeError), Stage(1, 1, "patch"):
        raise RuntimeError("boom")
    assert "error: [1/1] patch failed after" in console.getvalue()


def test_warnings_prefixed_and_quiet_mode() -> None:
    stream = io.StringIO()
    setup_logging(QUIET, stream=stream)
    get_logger("x").info("hidden")
    get_logger("x").warning("3 documents are empty")
    assert stream.getvalue() == "warning: 3 documents are empty\n"
    setup_logging(stream=io.StringIO())


def test_progress_non_tty_logs_at_each_decile(console: io.StringIO) -> None:
    with Progress(100, "[1/1] work", unit="files", interval=3600) as progress:
        assert not progress.tty
        for _ in range(100):
            progress.advance()
    lines = [line for line in console.getvalue().splitlines() if line.startswith("[1/1] work")]
    assert len(lines) == 10
    assert "100/100 files · 100%" in lines[-1]


def test_progress_tty_render() -> None:
    progress = Progress(200, "[3/5] blast", unit="jobs")
    progress.done = 50
    text = progress.render()
    assert "███░░░░░░░░░" in text and "50/200 jobs · 25%" in text and "elapsed" in text


def _work_in_worker(n: int) -> int:
    get_logger("worker").info(f"worker says {n}")
    return n


def test_worker_records_reach_main_process(console: io.StringIO) -> None:
    with LogListener() as listener, ProcessPoolExecutor(
        max_workers=2, initializer=configure_worker_logging, initargs=(listener.queue, logging.INFO)
    ) as pool:
        assert sorted(pool.map(_work_in_worker, [1, 2])) == [1, 2]
    output = console.getvalue()
    assert "worker says 1" in output and "worker says 2" in output
