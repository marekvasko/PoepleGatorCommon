"""Reusable, logging-based progress reporting with rate and ETA estimates."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sized
import logging
from time import monotonic
from types import TracebackType
from typing import TypeVar

T = TypeVar("T")


def _format_duration(seconds: float) -> str:
    """Format a non-negative duration compactly for progress logs."""
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


class LoggingProgress:
    """Report progress through a standard :class:`logging.Logger`.

    Updates are emitted no more often than ``min_interval_seconds``. The first
    and final states are always logged. Instances may be used as context
    managers so interrupted work is reported as stopped or failed. Every log
    record also exposes its values as ``progress_*`` attributes through
    :mod:`logging`'s ``extra`` mechanism.
    """

    def __init__(
        self,
        logger: logging.Logger,
        *,
        total: int | None = None,
        description: str = "Progress",
        unit: str = "items",
        min_interval_seconds: float = 30.0,
        level: int = logging.INFO,
        initial: int = 0,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if total is not None and total < 0:
            raise ValueError("total must be non-negative")
        if initial < 0:
            raise ValueError("initial must be non-negative")
        if total is not None and initial > total:
            raise ValueError("initial must not exceed total")
        if min_interval_seconds < 0:
            raise ValueError("min_interval_seconds must be non-negative")
        if not description:
            raise ValueError("description must not be empty")
        if not unit:
            raise ValueError("unit must not be empty")

        self.logger = logger
        self.total = total
        self.description = description
        self.unit = unit
        self.min_interval_seconds = min_interval_seconds
        self.level = level
        self.initial = initial
        self.completed = initial
        self._clock = clock
        self._started_at: float | None = None
        self._last_log_at: float | None = None
        self._last_logged_completed: int | None = None
        self._closed = False

    def start(self) -> "LoggingProgress":
        """Start timing and emit the initial progress state once."""
        if self._started_at is None:
            now = self._clock()
            self._started_at = now
            self._last_log_at = now
            self._log(now)
        return self

    def update(self, advance: int = 1) -> None:
        """Advance the completed count and log if the interval has elapsed."""
        if advance < 0:
            raise ValueError("advance must be non-negative")
        if self._closed:
            raise RuntimeError("cannot update closed progress")
        self.start()
        self.completed += advance
        now = self._clock()
        finished = self.total is not None and self.completed >= self.total
        interval_elapsed = (
            self._last_log_at is None
            or now - self._last_log_at >= self.min_interval_seconds
        )
        if finished or interval_elapsed:
            self._log(now)

    def close(self, *, failed: bool = False) -> None:
        """Emit a final state when needed and prevent further updates."""
        if self._closed:
            return
        self.start()
        now = self._clock()
        if failed:
            self._log(now, status="failed", force=True)
        elif self.total is None:
            self._log(now, status="complete", force=True)
        elif self.completed < self.total:
            self._log(now, status="stopped", force=True)
        elif self._last_logged_completed != self.completed:
            self._log(now, force=True)
        self._closed = True

    def __enter__(self) -> "LoggingProgress":
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        failed = exc_type is not None and not issubclass(exc_type, GeneratorExit)
        self.close(failed=failed)

    def _log(
        self,
        now: float,
        *,
        status: str | None = None,
        force: bool = False,
    ) -> None:
        if not force and self._last_logged_completed == self.completed:
            return
        assert self._started_at is not None
        elapsed = max(0.0, now - self._started_at)
        progressed = self.completed - self.initial
        rate = progressed / elapsed if elapsed > 0 else 0.0
        percent: float | None = None
        remaining: int | None = None
        eta_seconds: float | None = None

        if self.total is None:
            message = f"{self.description}: {self.completed} {self.unit}"
        else:
            percent = (
                100.0
                if self.total == 0
                else 100 * self.completed / self.total
            )
            remaining = max(0, self.total - self.completed)
            eta_seconds = remaining / rate if rate > 0 else None
            message = (
                f"{self.description}: {self.completed}/{self.total} {self.unit} "
                f"({percent:.1f}%)"
            )

        message += (
            f", elapsed={_format_duration(elapsed)}, "
            f"rate={rate:.2f} {self.unit}/s"
        )
        if self.total is not None:
            eta = (
                _format_duration(eta_seconds)
                if eta_seconds is not None
                else "unknown"
            )
            message += f", ETA={eta}"
        if status is not None:
            message += f", status={status}"

        effective_status = status or (
            "complete"
            if self.total is not None and self.completed >= self.total
            else "running"
        )
        extra: dict[str, object] = {
            "progress_description": self.description,
            "progress_completed": self.completed,
            "progress_total": self.total,
            "progress_percent": percent,
            "progress_elapsed_seconds": elapsed,
            "progress_rate": rate,
            "progress_eta_seconds": eta_seconds,
            "progress_remaining": remaining,
            "progress_unit": self.unit,
            "progress_status": effective_status,
        }
        self.logger.log(self.level, message, extra=extra)
        self._last_log_at = now
        self._last_logged_completed = self.completed


def log_progress(
    iterable: Iterable[T],
    *,
    logger: logging.Logger,
    total: int | None = None,
    description: str = "Progress",
    unit: str = "items",
    min_interval_seconds: float = 30.0,
    level: int = logging.INFO,
) -> Iterator[T]:
    """Yield ``iterable`` while periodically logging progress.

    ``total`` is inferred for sized iterables. Pass it explicitly for generators
    when ETA and percentage reporting are desired.
    """
    if total is None and isinstance(iterable, Sized):
        total = len(iterable)
    with LoggingProgress(
        logger,
        total=total,
        description=description,
        unit=unit,
        min_interval_seconds=min_interval_seconds,
        level=level,
    ) as progress:
        for item in iterable:
            yield item
            progress.update()


__all__ = ["LoggingProgress", "log_progress"]
