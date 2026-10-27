"""Worker pools are spawned on every platform (Linux's default "fork" can deadlock with Polars' threads)."""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl
import pytest

from textblast.parallel import process_pool, worker_polars_threads

SRC = Path(__file__).resolve().parents[1] / "src" / "textblast"


def _square(x: int) -> int:
    return x * x


def test_pool_spawns_and_runs() -> None:
    with process_pool(max_workers=2) as pool:
        assert pool._mp_context.get_start_method() == "spawn"
        assert list(pool.map(_square, range(4))) == [0, 1, 4, 9]


def test_every_pool_goes_through_the_helper() -> None:
    direct = [p.name for p in SRC.rglob("*.py")
              if p.name != "parallel.py" and "ProcessPoolExecutor(" in p.read_text(encoding="utf-8")]
    assert direct == []


def _polars_threads(_: int) -> tuple[str | None, int]:
    import os

    import polars as pl

    return os.environ.get("POLARS_MAX_THREADS"), pl.thread_pool_size()


def test_workers_get_a_share_of_the_cores(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POLARS_MAX_THREADS", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    assert worker_polars_threads(2) == 4 and worker_polars_threads(None) == 1 and worker_polars_threads(16) == 1
    parent_threads = pl.thread_pool_size()
    with process_pool(max_workers=2) as pool:
        assert set(pool.map(_polars_threads, range(2))) == {("4", 4)}
    assert "POLARS_MAX_THREADS" not in os.environ  # restored after the pool
    assert pl.thread_pool_size() == parent_threads  # the parent keeps its own pool


def test_a_given_polars_threads_setting_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLARS_MAX_THREADS", "3")
    with process_pool(max_workers=2) as pool:
        assert set(pool.map(_polars_threads, range(2))) == {("3", 3)}
    assert os.environ["POLARS_MAX_THREADS"] == "3"
