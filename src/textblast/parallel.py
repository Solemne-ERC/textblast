"""Worker processes: every pool starts fresh interpreters ("spawn"), on every platform.

Linux's default before Python 3.14 is "fork", which copies the parent with Polars' thread pool already running; Polars
warns that this can deadlock. "spawn" is macOS's and Windows' default, so all platforms behave the same.

Each spawned worker would start Polars with one thread per core, so N workers on N cores would run N² threads. While a
pool is open, its workers get ``POLARS_MAX_THREADS`` = cores ÷ workers (at least 1), unless the variable is already
set, in which case everyone uses the given value.
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

START_METHOD = "spawn"
POLARS_THREADS = "POLARS_MAX_THREADS"


def worker_polars_threads(max_workers: int | None, cores: int | None = None) -> int:
    cores = cores or os.cpu_count() or 1
    return max(1, cores // (max_workers or cores))


class _Pool(ProcessPoolExecutor):
    """Sets the workers' Polars threads while open: spawned workers read the variable when they import Polars."""

    def __init__(self, polars_threads: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._polars_threads = polars_threads
        self._set_variable = False

    def __enter__(self) -> _Pool:
        if POLARS_THREADS not in os.environ:
            import polars as pl

            pl.thread_pool_size()  # this process's own pool is created now, with all cores, before the variable is set
            os.environ[POLARS_THREADS] = str(self._polars_threads)
            self._set_variable = True
        return self

    def __exit__(self, *exc_info: Any) -> Any:
        try:
            return super().__exit__(*exc_info)  # waits for the workers
        finally:
            if self._set_variable:
                os.environ.pop(POLARS_THREADS, None)
                self._set_variable = False


def process_pool(max_workers: int | None = None, **kwargs: Any) -> ProcessPoolExecutor:
    """A ``ProcessPoolExecutor`` whose workers are spawned, not forked; use it with ``with``."""
    return _Pool(worker_polars_threads(max_workers), max_workers=max_workers,
                 mp_context=multiprocessing.get_context(START_METHOD), **kwargs)
