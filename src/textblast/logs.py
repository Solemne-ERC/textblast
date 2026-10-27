"""Logging and progress reporting.

The library only ever logs to the ``textblast`` logger; handlers are configured by the CLI via
:func:`setup_logging`. Console output stays short: numbered stage lines, one progress line, warnings and errors.
"""

from __future__ import annotations

import logging
import logging.handlers
import multiprocessing
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import IO, Any

LOGGER_NAME = "textblast"
logger = logging.getLogger(LOGGER_NAME)

QUIET, NORMAL, VERBOSE = -1, 0, 1

_FILE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

# Console state, set by setup_logging(); the library itself never configures handlers.
_console_handler: ConsoleHandler | None = None
_active_progress: Progress | None = None


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the package logger or one of its children (``textblast.<name>``)."""
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)


# --------------------------------------------------------------------------- formatting helpers


def format_duration(seconds: float) -> str:
    """Compact duration: ``3.2s``, ``42s``, ``7m``, ``25m``, ``1h12m``."""
    if seconds < 10:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    if minutes < 60:
        rest = int(seconds % 60)
        return f"{minutes}m" if minutes >= 10 else f"{minutes}m{rest:02d}s"
    return f"{minutes // 60}h{minutes % 60:02d}m"


def format_bytes(n: float) -> str:
    """Human-readable byte count with binary units (``12.3 MB``)."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"  # pragma: no cover


def format_count(n: float) -> str:
    """Compact count: ``950``, ``12.3k``, ``4.2M``, ``1.5G``, ``3.1T``."""
    if n < 1000:
        return f"{n:.0f}"
    for divisor, suffix in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if n >= divisor:
            return f"{n / divisor:.1f}{suffix}"
    return f"{n:.0f}"  # pragma: no cover


def peak_memory_bytes() -> int | None:
    """Peak resident memory of this process or any finished child, if the platform reports it."""
    try:
        import resource
    except ImportError:  # Windows
        return None
    usage = max(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    )
    return usage if sys.platform == "darwin" else usage * 1024  # macOS reports bytes, Linux KiB


# --------------------------------------------------------------------------- handlers and setup


class ConsoleFormatter(logging.Formatter):
    """Plain messages; warnings and errors are prefixed; tracebacks only when verbose."""

    def __init__(self, show_tracebacks: bool) -> None:
        super().__init__()
        self.show_tracebacks = show_tracebacks

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if record.levelno >= logging.ERROR:
            message = f"error: {message}"
        elif record.levelno >= logging.WARNING:
            message = f"warning: {message}"
        if self.show_tracebacks and record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"
        return message


class ConsoleHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """Writes to stderr and keeps an active in-place progress line intact."""

    def emit(self, record: logging.LogRecord) -> None:
        progress = _active_progress
        if progress is not None and progress.tty:
            progress.clear()
        super().emit(record)
        if progress is not None and progress.tty:
            progress.redraw()


def setup_logging(verbosity: int = NORMAL, log_file: Path | None = None, stream: IO[str] | None = None) -> None:
    """Configure console (and optionally file) output for the CLI.

    ``verbosity``: ``QUIET`` shows warnings and errors only, ``NORMAL`` adds stage and progress lines,
    ``VERBOSE`` adds debug detail and tracebacks. Calling it again replaces the previous configuration.
    """
    global _console_handler
    for handler in list(logger.handlers):
        if getattr(handler, "_textblast", False):
            logger.removeHandler(handler)
            handler.close()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    console = ConsoleHandler(stream if stream is not None else sys.stderr)
    console.setLevel({QUIET: logging.WARNING, NORMAL: logging.INFO}.get(verbosity, logging.DEBUG))
    console.setFormatter(ConsoleFormatter(show_tracebacks=verbosity >= VERBOSE))
    console._textblast = True  # type: ignore[attr-defined]
    logger.addHandler(console)
    _console_handler = console

    if log_file is not None:
        add_log_file(log_file)


def add_log_file(path: Path) -> logging.Handler:
    """Log everything (debug level, with timestamps and tracebacks) to ``path`` in addition to the console."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(_FILE_FORMAT))
    handler._textblast = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    return handler


def remove_handler(handler: logging.Handler) -> None:
    logger.removeHandler(handler)
    handler.close()


def _console_shows(level: int) -> bool:
    return _console_handler is not None and _console_handler.level <= level


# --------------------------------------------------------------------------- stages


class Stage:
    """A numbered pipeline stage: ``[3/5] name: detail`` at start, ``[3/5] name done in 25m: summary`` at end.

    Use as a context manager; set :attr:`summary` before leaving to report the stage's main numbers.
    """

    def __init__(
        self,
        index: int,
        total: int,
        name: str,
        detail: str = "",
        log: logging.Logger | None = None,
        show_memory: bool = False,
    ) -> None:
        self.index, self.total, self.name, self.detail = index, total, name, detail
        self.log = log or logger
        self.show_memory = show_memory
        self.summary = ""
        self.started = 0.0
        self.elapsed = 0.0

    @property
    def label(self) -> str:
        return f"[{self.index}/{self.total}] {self.name}"

    def __enter__(self) -> Stage:
        self.started = time.monotonic()
        self.log.info(f"{self.label}: {self.detail}" if self.detail else self.label)
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.elapsed = time.monotonic() - self.started
        if exc_type is not None:
            self.log.error(f"{self.label} failed after {format_duration(self.elapsed)}")
            return
        parts = [f"{self.label} done in {format_duration(self.elapsed)}"]
        extra = [self.summary] if self.summary else []
        if self.show_memory and (peak := peak_memory_bytes()):
            extra.append(f"peak memory {format_bytes(peak)}")
        self.log.info(": ".join(parts + [" · ".join(extra)]) if extra else parts[0])

    def progress(
        self, total: float | None, unit: str = "", formatter: Callable[[float], str] | None = None
    ) -> Progress:
        """A progress line labelled with this stage."""
        return Progress(total, self.label, unit=unit, formatter=formatter, log=self.log)


# --------------------------------------------------------------------------- progress


class Progress:
    """One progress line measured in units of work, with elapsed time and a remaining-time estimate.

    On a terminal the line updates in place (at most a few times per second). Otherwise a plain line is logged
    every ``interval`` seconds and whenever another 10% is completed. ``total=None`` shows only the amount done.
    """

    BAR_WIDTH = 12

    def __init__(
        self,
        total: float | None,
        label: str,
        unit: str = "",
        formatter: Callable[[float], str] | None = None,
        log: logging.Logger | None = None,
        interval: float = 60.0,
        refresh: float = 0.2,
        stream: IO[str] | None = None,
    ) -> None:
        self.total = total if total else None
        self.label, self.unit = label, unit
        self.formatter = formatter or format_count
        self.log = log or logger
        self.interval, self.refresh = interval, refresh
        self.done = 0.0
        self.note = ""  # short extra text, e.g. "21/30 jobs"
        self.started = time.monotonic()
        self._last_draw = 0.0
        self._last_line = self.started
        self._last_decile = 0
        self._visible = False
        console_stream = getattr(_console_handler, "stream", None)
        self.stream = stream or console_stream or sys.stderr
        self.enabled = _console_shows(logging.INFO)
        self.tty = self.enabled and bool(getattr(self.stream, "isatty", lambda: False)())

    # context manager
    def __enter__(self) -> Progress:
        global _active_progress
        _active_progress = self
        self.redraw(force=True)
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def advance(self, amount: float = 1.0) -> None:
        self.set(self.done + amount)

    def set(self, done: float, note: str | None = None) -> None:
        self.done = done
        if note is not None:
            self.note = note
        now = time.monotonic()
        if self.tty:
            if now - self._last_draw >= self.refresh:
                self.redraw()
            return
        decile = int(10 * self.done / self.total) if self.total else 0
        if now - self._last_line >= self.interval or decile > self._last_decile:
            self._last_line, self._last_decile = now, max(decile, self._last_decile)
            if self.enabled:
                self.log.info(self.render(bar=False))

    def render(self, bar: bool = True) -> str:
        elapsed = time.monotonic() - self.started
        amount = self.formatter(self.done)
        unit = f" {self.unit}" if self.unit else ""
        if not self.total:
            extra = f" · {self.note}" if self.note else ""
            return f"{self.label} {amount}{unit}{extra} · {format_duration(elapsed)} elapsed"
        fraction = min(self.done / self.total, 1.0)
        parts = [self.label]
        if bar:
            filled = int(round(fraction * self.BAR_WIDTH))
            parts.append("█" * filled + "░" * (self.BAR_WIDTH - filled))
        if self.note:
            parts.append(self.note + " ·")
        parts.append(f"{amount}/{self.formatter(self.total)}{unit} · {fraction:.0%}")
        line = " ".join(parts) + f" · {format_duration(elapsed)} elapsed"
        if 0.02 < fraction < 1.0 and elapsed > 2:
            line += f" · ~{format_duration(elapsed * (1 - fraction) / fraction)} left"
        return line

    def clear(self) -> None:
        if self.tty and self._visible:
            self.stream.write("\r\033[K")
            self.stream.flush()
            self._visible = False

    def redraw(self, force: bool = False) -> None:
        if not self.tty:
            return
        self.stream.write("\r\033[K" + self.render())
        self.stream.flush()
        self._visible = True
        self._last_draw = time.monotonic()

    def close(self) -> None:
        global _active_progress
        self.clear()
        if _active_progress is self:
            _active_progress = None


# --------------------------------------------------------------------------- worker processes


class LogListener:
    """Forwards log records from worker processes to the main process's handlers.

    Usage::

        with LogListener() as listener:
            with process_pool(initializer=configure_worker_logging, initargs=(listener.queue,)) as pool:
                ...
    """

    def __init__(self) -> None:
        self.queue: Any = multiprocessing.get_context("spawn").Queue()
        self._listener: logging.handlers.QueueListener | None = None

    def __enter__(self) -> LogListener:
        self._listener = logging.handlers.QueueListener(
            self.queue, _ForwardToLogger(), respect_handler_level=False
        )
        self._listener.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None


class _ForwardToLogger(logging.Handler):
    """Hands records received from workers to the main process's logger (and thus its handlers)."""

    def emit(self, record: logging.LogRecord) -> None:
        logging.getLogger(record.name).handle(record)


def configure_worker_logging(queue: Any, level: int = logging.DEBUG) -> None:
    """Process-pool initializer: send this worker's ``textblast`` records to ``queue``."""
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.addHandler(logging.handlers.QueueHandler(queue))
    logger.setLevel(level)
    logger.propagate = False
