from __future__ import annotations
from collections.abc import Sequence
import logging
from pathlib import Path
from requests import Session
from people_gator_common.rate_limiter import RateLimiter
from people_gator_common.crawling.fetchers import (
    download_missing as _download_missing,
    BatchDownloadStats
)
from urllib.parse import urlparse, unquote


def derive_image_path(url: str, save_root: Path | str) -> Path | str | None:
    logger = logging.getLogger(__name__ + "." + derive_image_path.__name__)
    parsed_url = urlparse(url)
    if not parsed_url.scheme.startswith("http"):
        logger.warning("Skipping non-http URL: %s", url)
        return None
    return Path(save_root) / parsed_url.netloc / Path(unquote(parsed_url.path.lstrip("/")))


def download_missing_images(
    urls: Sequence[str],
    output_dir: Path | str,
    *,
    session: Session,
    rate_limiter: RateLimiter,
    show_progress: bool = True,
 ) -> BatchDownloadStats:
    stats = _download_missing(
        urls,
        output_dir=output_dir,
        path_func=derive_image_path,
        session=session,
        rate_limiter=rate_limiter,
        mode="binary",
        show_progress=show_progress,
    )
    return stats
