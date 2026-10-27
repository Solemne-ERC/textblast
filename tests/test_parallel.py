"""Worker pools are spawned on every platform (Linux's default "fork" can deadlock with Polars' threads)."""

from __future__ import annotations

from pathlib import Path

from textblast.parallel import process_pool

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
