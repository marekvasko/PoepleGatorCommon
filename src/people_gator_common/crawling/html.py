from __future__ import annotations
from collections.abc import Sequence
import logging
import os
from pathlib import Path
from requests import Session
from people_gator_common.rate_limiter import RateLimiter
from people_gator_common.crawling.fetchers import (
    download_missing,
    BatchDownloadStats
)
from urllib.parse import urlparse, unquote


def derive_html_path(url: str, save_root: str) -> str | None:
    logger = logging.getLogger(__name__ + "." + derive_html_path.__name__)
    parsed_url = urlparse(url)
    if not parsed_url.scheme.startswith("http"):
        logger.warning("Skipping non-http URL: %s", url)
        return None
    return os.path.join(
        save_root, 
        parsed_url.netloc, 
        unquote(parsed_url.path.lstrip("/")), 
        f"{unquote(parsed_url.query)}.html"
    )


def download_missing_html(
    urls: Sequence[str],
    output_dir: str,
    *,
    session: Session,
    rate_limiter: RateLimiter,
    show_progress: bool = True,
 ) -> BatchDownloadStats:
    stats = download_missing(
        urls,
        output_dir=output_dir,
        path_func=derive_html_path,
        session=session,
        rate_limiter=rate_limiter,
        mode="text",
        show_progress=show_progress,
    )
    return stats
