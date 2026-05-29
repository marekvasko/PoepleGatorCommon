from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import pandas as pd
from requests import Session
from tqdm.contrib.logging import logging_redirect_tqdm

from people_gator_common.headers import CLIENT_HEADERS
from people_gator_common.transport.images import download_missing_images
from people_gator_common.transport.fetchers import BatchDownloadStats
from people_gator_common.rate_limiter import RateLimiter
from people_gator_common.filesystem import open_file

def _extract_image_urls_from_csv(image_csv: str, *, url_column: str) -> list[str]:
    with open_file(image_csv, "r") as f:
        images_df = pd.read_csv(f)
    if url_column not in images_df.columns:
        raise ValueError(f"CSV file {image_csv} must contain '{url_column}' column")
    urls = images_df[url_column].dropna().astype(str).str.strip()
    urls = urls[urls != ""]
    return urls.tolist()


def run_wikicommons_images_download_pipeline(
    input_csv: str,
    output_dir: str,
    *,
    rate_limiter: RateLimiter,
    url_column: str = "value",
    show_progress: bool = True,
) -> BatchDownloadStats:
    image_urls = _extract_image_urls_from_csv(input_csv, url_column=url_column)
    progress_context = logging_redirect_tqdm() if show_progress else nullcontext()
    with progress_context, Session() as session:
        session.headers.update(CLIENT_HEADERS)
        return download_missing_images(
            urls=image_urls,
            output_dir=output_dir,
            session=session,
            rate_limiter=rate_limiter,
            show_progress=show_progress,
        )
