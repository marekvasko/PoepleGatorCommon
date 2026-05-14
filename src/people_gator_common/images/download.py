from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
import logging
from pathlib import Path
from urllib.parse import unquote, urlparse

import tqdm
from requests import Session
from people_gator_common.rate_limiter import RateLimiter, RateLimiterTooManyRequestsError
from people_gator_common.errors import is_retryable_error


def get_save_path(url: str, *, save_root: Path) -> Path | None:
    logger = logging.getLogger(__name__ + "." + get_save_path.__name__)

    parsed_url = urlparse(url)
    if not parsed_url.scheme.startswith("http"):
        logger.warning("Skipping non-http URL: %s", url)
        return None

    source = parsed_url.netloc
    image_path = Path(unquote(parsed_url.path).lstrip("/"))
    return save_root / source / image_path


def download_image(
    url: str,
    output_dir: Path,
    *,
    session: Session,
    rate_limiter: RateLimiter,
) -> bool:
    logger = logging.getLogger(__name__ + "." + download_image.__name__)

    save_path = get_save_path(url, save_root=output_dir)
    if save_path is None:
        return False

    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists():
        logger.info("Image already exists, skipping download: %s", save_path)
        return True

    # Use the provided RateLimiter instance for spacing and attempt tracking
    with rate_limiter as rl:
        while True:
            try:
                rl.request()
                response = session.get(url, timeout=30)
                response.raise_for_status()
                with open(save_path, "wb") as file_handle:
                    file_handle.write(response.content)
                logger.info("Downloaded image from %s to %s", url, save_path)
                return True
            except RateLimiterTooManyRequestsError as error:
                logger.warning("Too many requests error: %s", error)
                return False
            except Exception as error:
                logger.warning("Error: %s", error)
                if is_retryable_error(error):
                    logger.info("Retrying after error")
                    continue
                return False


@dataclass(frozen=True, slots=True)
class ImageBatchDownloadStats:
    total: int
    queued: int
    skipped_existing: int
    downloaded: int
    failed: int


def _split_urls_by_existing_files(urls: Sequence[str], output_dir: Path) -> tuple[list[str], int]:
    urls_to_download: list[str] = []
    skipped_existing = 0
    for url in urls:
        save_path = get_save_path(url, save_root=output_dir)
        if save_path is not None and save_path.exists():
            skipped_existing += 1
            continue
        urls_to_download.append(url)
    return urls_to_download, skipped_existing


def download_missing_images(
    urls: Sequence[str],
    output_dir: Path,
    *,
    session: Session,
    rate_limiter: RateLimiter,
    show_progress: bool = True,
) -> ImageBatchDownloadStats:
    urls_to_download, skipped_existing = _split_urls_by_existing_files(urls, output_dir)
    downloaded = 0
    failed = 0

    iterator: Sequence[str] | tqdm.tqdm[str]
    if show_progress:
        iterator = tqdm.tqdm(urls_to_download, total=len(urls_to_download), desc="Downloading images")
    else:
        iterator = urls_to_download

    for image_url in iterator:
        success = download_image(
            image_url,
            output_dir=output_dir,
            rate_limiter=rate_limiter,
            session=session,
        )
        if success:
            downloaded += 1
        else:
            failed += 1

    return ImageBatchDownloadStats(
        total=len(urls),
        queued=len(urls_to_download),
        skipped_existing=skipped_existing,
        downloaded=downloaded,
        failed=failed,
    )
