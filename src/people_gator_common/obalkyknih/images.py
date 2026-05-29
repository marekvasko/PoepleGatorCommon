"""ObalkyKnih image download utilities."""

from __future__ import annotations

import logging
from os.path import join
from urllib.parse import urlparse

from requests import Session

from people_gator_common.transport.fetchers import BatchDownloadStats, download_missing
from people_gator_common.headers import CLIENT_HEADERS
from people_gator_common.rate_limiter import RateLimiter
from people_gator_common.utils import read_column_from_csv


def obalkyknih_image_path_func(url: str, output_dir: str) -> str | None:
    """Derive the local save path for an ObalkyKnih preview image URL."""
    logger = logging.getLogger(__name__ + "." + obalkyknih_image_path_func.__name__)
    if not url.startswith("http"):
        logger.warning("Skipping non-http URL: %s", url)
        return None
    parsed_url = urlparse(url)
    netloc = parsed_url.netloc.replace(":", "_")
    split_path = parsed_url.path.lstrip("/").split("/")
    path = join(*split_path[:-1]) + "_" + split_path[-1] if len(split_path) > 1 else split_path[0]
    return join(output_dir, netloc, f"{path}.png")


def download_missing_obalkyknih_images(
    input_csv: str,
    output_dir: str,
    *,
    rate_limiter: RateLimiter,
    url_column: str = "value",
    show_progress: bool = True,
) -> BatchDownloadStats:
    """Download ObalkyKnih preview images listed in *input_csv*."""
    urls = read_column_from_csv(input_csv, url_column)
    with Session() as session:
        session.headers.update(CLIENT_HEADERS)
        return download_missing(
            urls=urls,
            output_dir=output_dir,
            path_func=obalkyknih_image_path_func,
            session=session,
            rate_limiter=rate_limiter,
            mode="binary",
            show_progress=show_progress,
        )
