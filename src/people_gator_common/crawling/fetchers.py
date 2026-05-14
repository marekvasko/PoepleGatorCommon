from __future__ import annotations
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal
from requests import Session
from people_gator_common.rate_limiter import RateLimiter, RateLimiterTooManyRequestsError
from people_gator_common.errors import is_retryable_error
import logging
import tqdm

type PathFunc = Callable[[str, Path], Path | None]


from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BatchDownloadStats:
    total: int
    queued: int
    skipped_existing: int
    downloaded: int
    failed: int


def split_urls_by_existing_files(urls: Sequence[str], output_dir: Path, path_func: PathFunc) -> tuple[list[str], int]:
    urls_to_download: list[str] = []
    skipped_existing = 0
    for url in urls:
        save_path = path_func(url, output_dir)
        if save_path is not None and save_path.exists():
            skipped_existing += 1
            continue
        urls_to_download.append(url)
    return urls_to_download, skipped_existing


def download_url(
    url: str,
    output_dir: Path,
    *,
    path_func: PathFunc,
    session: Session,
    rate_limiter: RateLimiter,
    mode: Literal["text", "binary"] = "binary",
) -> bool:
    logger = logging.getLogger(__name__ + "." + download_url.__name__)
    save_path = path_func(url, output_dir)
    if save_path is None:
        return False
    if save_path.exists():
        logger.info(
            "File already exists, skipping download for url %s: %s", url, save_path,
            extra={"url": url, "save_path": str(save_path)},
        )
        return True

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with rate_limiter as rl:
        while True:
            try:
                rl.request()
                response = session.get(url)
                response.raise_for_status()
                if mode == "text":
                    resp_len = len(response.text)
                elif mode == "binary":
                    resp_len = len(response.content)
                else:
                    raise ValueError(f"Invalid mode: {mode}")
                logger.info(
                    "Successfully fetched from url %s",
                    url,
                    extra={"url": url, "response_length": resp_len},
                )
                
                if mode == "text":
                    with open(save_path, "w", encoding="utf-8") as fh:
                        fh.write(response.text)
                elif mode == "binary":
                    with open(save_path, "wb") as fh:
                        fh.write(response.content)
                else:
                    raise ValueError(f"Invalid mode: {mode}")
                logger.info(
                    "Saved file for url %s to %s",
                    url,
                    save_path,
                    extra={"url": url, "save_path": str(save_path)},
                )
                return True
            except RateLimiterTooManyRequestsError as error:
                logger.warning(
                    "Too many requests error for url %s: %s", url, error, extra={"url": url}
                )
                return False
            except Exception as error:
                logger.warning(
                    "Error fetching url %s: %s", url, error, extra={"url": url}
                )
                if is_retryable_error(error):
                    logger.info("Retrying after error for url %s", url, extra={"url": url})
                    continue
                return False


def download_missing(
    urls: Sequence[str],
    output_dir: Path,
    *,
    path_func: PathFunc,
    session: Session,
    rate_limiter: RateLimiter,
    mode: Literal["text", "binary"] = "binary",
    show_progress: bool = True,
) -> BatchDownloadStats:
    logger = logging.getLogger(__name__ + "." + download_missing.__name__)
    logger.info("Splitting URLs by existing files in output directory: %s", output_dir)
    urls_to_download, skipped_existing = split_urls_by_existing_files(urls, output_dir, path_func)
    logger.info(
        "Split URLs into %d to download and %d existing files to skip",
        len(urls_to_download),
        skipped_existing,
        extra={
            "total_urls": len(urls), 
            "urls_to_download": len(urls_to_download), 
            "skipped_existing": skipped_existing
        },
    )
    
    downloaded = 0
    failed = 0

    iterator: Sequence[str] | tqdm.tqdm[str]
    if show_progress:
        iterator = tqdm.tqdm(urls_to_download, total=len(urls_to_download), desc="Downloading")
    else:
        iterator = urls_to_download

    for url in iterator:
        success = download_url(
            url,
            output_dir=output_dir,
            path_func=path_func,
            rate_limiter=rate_limiter,
            session=session,
            mode=mode,
        )
        if success:
            downloaded += 1
        else:
            failed += 1

    return BatchDownloadStats(
        total=len(urls),
        queued=len(urls_to_download),
        skipped_existing=skipped_existing,
        downloaded=downloaded,
        failed=failed,
    )
