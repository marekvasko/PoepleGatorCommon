from contextlib import nullcontext
import logging
import os

import pandas as pd
from pathlib import Path

from requests import Session
from people_gator_common.transport import download_missing_html, BatchDownloadStats
from people_gator_common.filesystem import open_file
from people_gator_common.headers import CLIENT_HEADERS
from people_gator_common.rate_limiter import RateLimiter
from tqdm.contrib.logging import logging_redirect_tqdm

OBALKYKNIH_URL_TEMPLATE = "https://obalkyknih.cz/view_auth?auth_id={auth_id}"

def generate_obalkyknih_url(nlcr_aut_id: str) -> str:
    return OBALKYKNIH_URL_TEMPLATE.format(auth_id=nlcr_aut_id)

def obalkyknih_urls_from_csv(
    input_csv: str,
    *,
    auth_id_column: str,
) -> list[str]:
    with open_file(input_csv, "r") as f:
        df = pd.read_csv(f)
    if auth_id_column not in df.columns:
        raise ValueError(f"CSV file {input_csv} must contain '{auth_id_column}' column")
    auth_ids = df[auth_id_column].dropna().astype(str).str.strip()
    auth_ids = auth_ids[auth_ids != ""]
    urls = [generate_obalkyknih_url(auth_id) for auth_id in auth_ids]
    return urls

def download_missing_obalkyknih_htmls(
    input_csv: str,
    output_dir: str,
    *,
    rate_limiter: RateLimiter,
    auth_id_column: str = "entity",
    show_progress: bool = True,
) -> BatchDownloadStats:
    logger = logging.getLogger(__name__ + "." + download_missing_obalkyknih_htmls.__name__)
    
    logger.info("Generating ObalkyKnih HTML URLs from CSV: %s", input_csv)
    html_urls = obalkyknih_urls_from_csv(input_csv, auth_id_column=auth_id_column)
    logger.info("Generated %d ObalkyKnih HTML URLs", len(html_urls))
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
