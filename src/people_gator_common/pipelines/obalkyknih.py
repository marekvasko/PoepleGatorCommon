from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import pandas as pd
from requests import Session
from tqdm.contrib.logging import logging_redirect_tqdm

from people_gator_common.headers import CLIENT_HEADERS
from people_gator_common.crawling.html import download_missing_html
from people_gator_common.crawling.fetchers import BatchDownloadStats
from people_gator_common.rate_limiter import RateLimiter

OBALKYKNIH_URL_TEMPLATE = "https://obalkyknih.cz/view_auth?auth_id={auth_id}"

def generate_obalkyknih_url(nlcr_aut_id: str) -> str:
    return OBALKYKNIH_URL_TEMPLATE.format(auth_id=nlcr_aut_id)

def _generate_html_urls_from_csv(
    input_csv: Path,
    *,
    auth_id_column: str,
) -> list[str]:
    df = pd.read_csv(input_csv)
    if auth_id_column not in df.columns:
        raise ValueError(f"CSV file {input_csv} must contain '{auth_id_column}' column")
    auth_ids = df[auth_id_column].dropna().astype(str).str.strip()
    auth_ids = auth_ids[auth_ids != ""]
    urls = [generate_obalkyknih_url(auth_id) for auth_id in auth_ids]
    return urls

def run_obalkyknih_html_download_pipeline(
    input_csv: Path,
    output_dir: Path,
    *,
    rate_limiter: RateLimiter,
    auth_id_column: str = "auth_id",
    show_progress: bool = True,
) -> BatchDownloadStats:
    html_urls = _generate_html_urls_from_csv(input_csv, auth_id_column=auth_id_column)
    progress_context = logging_redirect_tqdm() if show_progress else nullcontext()
    with progress_context, Session() as session:
        session.headers.update(CLIENT_HEADERS)
        return download_missing_html(
            urls=html_urls,
            output_dir=output_dir,
            session=session,
            rate_limiter=rate_limiter,
            show_progress=show_progress,
        )
