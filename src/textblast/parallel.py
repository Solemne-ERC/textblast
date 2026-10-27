"""Worker processes: every pool starts fresh interpreters ("spawn"), on every platform.

Linux's default before Python 3.14 is "fork", which copies the parent with Polars' thread pool already running; Polars
warns that this can deadlock. "spawn" is macOS's and Windows' default, so all platforms behave the same.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Any

START_METHOD = "spawn"


def process_pool(max_workers: int | None = None, **kwargs: Any) -> ProcessPoolExecutor:
    """A ``ProcessPoolExecutor`` whose workers are spawned, not forked."""
    return ProcessPoolExecutor(max_workers=max_workers, mp_context=multiprocessing.get_context(START_METHOD), **kwargs)
