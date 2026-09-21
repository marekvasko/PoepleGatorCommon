from __future__ import annotations

import logging

from people_gator_common.progress import LoggingProgress, log_progress


class RecordingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _recording_logger() -> tuple[logging.Logger, RecordingHandler]:
    logger = logging.getLogger(f"test.progress.{id(object())}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = RecordingHandler()
    logger.handlers = [handler]
    return logger, handler


def test_logging_progress_throttles_and_adds_structured_extra() -> None:
    logger, handler = _recording_logger()
    clock = FakeClock()

    with LoggingProgress(
        logger,
        total=4,
        description="Clustering",
        unit="pages",
        min_interval_seconds=10,
        clock=clock,
    ) as progress:
        progress.update()
        clock.advance(10)
        progress.update()
        clock.advance(10)
        progress.update(2)

    assert len(handler.records) == 3
    middle = handler.records[1]
    assert middle.getMessage().startswith("Clustering: 2/4 pages (50.0%)")
    assert middle.progress_description == "Clustering"
    assert middle.progress_completed == 2
    assert middle.progress_total == 4
    assert middle.progress_percent == 50.0
    assert middle.progress_elapsed_seconds == 10.0
    assert middle.progress_rate == 0.2
    assert middle.progress_eta_seconds == 10.0
    assert middle.progress_remaining == 2
    assert middle.progress_unit == "pages"
    assert middle.progress_status == "running"

    final = handler.records[-1]
    assert "ETA=0s" in final.getMessage()
    assert final.progress_completed == 4
    assert final.progress_status == "complete"


def test_log_progress_infers_total_for_sized_iterables() -> None:
    logger, handler = _recording_logger()

    result = list(
        log_progress(
            ["a", "b"],
            logger=logger,
            min_interval_seconds=0,
        )
    )

    assert result == ["a", "b"]
    assert handler.records[-1].progress_total == 2
    assert handler.records[-1].progress_completed == 2
    assert handler.records[-1].progress_status == "complete"


def test_log_progress_reports_early_close_as_stopped() -> None:
    logger, handler = _recording_logger()
    iterator = log_progress(
        range(3),
        logger=logger,
        min_interval_seconds=30,
    )

    assert next(iterator) == 0
    iterator.close()
    assert handler.records[-1].progress_status == "stopped"
