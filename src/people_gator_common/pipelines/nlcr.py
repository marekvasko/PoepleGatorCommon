from __future__ import annotations

from pathlib import Path

from requests import Session

from people_gator_common.headers import CLIENT_HEADERS
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
    nlcr_aut_ids = extract_nlcr_aut_ids_from_csv(
        csv_path=input_csv,
        id_column=id_column,
        deduplicate=deduplicate_ids,
        limit=limit,
    )

    with Session() as session:
        session.headers.update(CLIENT_HEADERS)
        return download_obalkyknih_html_batch(
            nlcr_aut_ids=nlcr_aut_ids,
            output_dir=output_dir,
            session=session,
            rate_limiter=rate_limiter,
            show_progress=show_progress,
        )
