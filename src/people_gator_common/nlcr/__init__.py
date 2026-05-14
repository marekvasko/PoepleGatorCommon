from __future__ import annotations

from pathlib import Path

from people_gator_common.nlcr.fetchers import NLCRDownloadStats, download_obalkyknih_html_batch
from people_gator_common.nlcr.processors import extract_nlcr_aut_ids_from_csv
from people_gator_common.rate_limiter import RateLimiter


def run_obalkyknih_html_pipeline(
    input_csv: Path,
    output_dir: Path,
    *,
    id_column: str = "entity",
    deduplicate_ids: bool = True,
    limit: int | None = None,
    rate_limiter: RateLimiter,
    show_progress: bool = True,
) -> NLCRDownloadStats:
    from people_gator_common.pipelines.nlcr import run_obalkyknih_html_pipeline as run_pipeline

    return run_pipeline(
        input_csv=input_csv,
        output_dir=output_dir,
        id_column=id_column,
        deduplicate_ids=deduplicate_ids,
        limit=limit,
        rate_limiter=rate_limiter,
        show_progress=show_progress,
    )

__all__ = [
    "NLCRDownloadStats",
    "download_obalkyknih_html_batch",
    "extract_nlcr_aut_ids_from_csv",
    "run_obalkyknih_html_pipeline",
]
