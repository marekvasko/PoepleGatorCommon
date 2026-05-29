from __future__ import annotations
from collections.abc import Callable, Sequence
import os
from typing import Literal
from requests import Session
from people_gator_common.rate_limiter import RateLimiter, RateLimiterInstance, RateLimiterTooManyRequestsError
from people_gator_common.errors import is_retryable_error
from people_gator_common.filesystem import ensure_parent_dir, glob, write_text, write_bytes
import logging
import tqdm
from dataclasses import dataclass
import mimetypes

type PathFunc = Callable[[str, str], str | None]

@dataclass(frozen=True, slots=True)
class BatchDownloadStats:
    total: int
    queued: int
    skipped_existing: int
    downloaded: int
    failed: int

def get_existing_files_in_output_dir(output_dir: str) -> set[str]:
    glob_pattern = os.path.join(str(output_dir), "**", "*")
    files = set(glob(glob_pattern))
    return files

def get_expected_file_locations(urls: Sequence[str], output_dir: str, path_func: PathFunc) -> list[str]:
    expected_files = []
    for url in urls:
        save_path = path_func(url, output_dir)
        if save_path is not None:
            expected_files.append(save_path)
    return expected_files

def split_urls_by_existing_files(
    urls: Sequence[str], 
    output_dir: str,
    path_func: PathFunc,
    *,
    show_progress: bool = True,
) -> tuple[list[str], list[str], int]:
    logger = logging.getLogger(__name__ + "." + split_urls_by_existing_files.__name__)
    expected_files = get_expected_file_locations(urls, output_dir, path_func)
    existing_files = get_existing_files_in_output_dir(output_dir)
    urls_to_download = []
    save_paths = []
    if show_progress:
        iterator = tqdm.tqdm(zip(urls, expected_files), total=len(urls), desc="Checking existing files")
    else:
        iterator = zip(urls, expected_files)
    
    for url, expected_file in iterator:
        if expected_file is None:
            logger.warning(
                "No valid save path for url %s, skipping download", 
                url, 
                extra={"url": url}
            )
            continue
        if expected_file in existing_files:
            logger.debug(
                "File already exists for url %s at expected path %s, skipping download", 
                url, expected_file, 
                extra={"url": url, "expected_file": expected_file}
            )
            continue
        urls_to_download.append(url)
        save_paths.append(expected_file)
        
    return urls_to_download, save_paths, len(urls_to_download)

def enforce_path_extension_from_mimetype(path: str | os.PathLike, mimetype: str) -> str:
    extension = mimetypes.guess_extension(mimetype)
    if extension is None:
        return str(path)
    if isinstance(path, str):
        if not path.lower().endswith(f".{extension}"):
            return f"{path}.{extension}"
    elif isinstance(path, os.PathLike):
        path_str = str(path)
        if not path_str.lower().endswith(f".{extension}"):
            return f"{path_str}.{extension}"
    return str(path)

def single_download_attempt(
    url: str,
    output_file: str,
    *,
    session: Session,
    rate_limiter_instance: RateLimiterInstance,
    mode: Literal["text", "binary"] = "binary",
) -> bool:
    rate_limiter_instance.request()
    logger = logging.getLogger(__name__ + "." + single_download_attempt.__name__)
    logger.info(
        "Attempting to download url %s with mode %s",
        url,
        mode,
        extra={"url": url, "mode": mode},
    )
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
        write_text(output_file, response.text)
    elif mode == "binary":
        write_bytes(output_file, response.content)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    logger.info(
        "Saved file for url %s to %s",
        url,
        output_file,
        extra={"url": url, "output_file": str(output_file)},
    )
    return True

def download_url(
    url: str,
    output_file: str,
    *,
    session: Session,
    rate_limiter: RateLimiter,
    mode: Literal["text", "binary"] = "binary",
) -> bool:
    logger = logging.getLogger(__name__ + "." + download_url.__name__)
    ensure_parent_dir(output_file)
    with rate_limiter as rl:
        while True:
            try:
                return single_download_attempt(
                    url,
                    output_file=output_file,
                    session=session,
                    rate_limiter_instance=rl,
                    mode=mode,
                )
            except RateLimiterTooManyRequestsError as error:
                logger.warning(
                    "Too many requests error for url %s: %s", 
                    url, error, 
                    extra={"url": url}
                )
                return False
            except Exception as error:
                if is_retryable_error(error):
                    logger.info(
                        "Retrying after error for url %s", 
                        url, 
                        extra={"url": url}
                    )
                    continue
                logger.warning(
                    "Error fetching url %s: %s", 
                    url, error, 
                    extra={"url": url}
                )
                return False


def download_missing(
    urls: Sequence[str],
    output_dir: str,
    *,
    path_func: PathFunc,
    session: Session,
    rate_limiter: RateLimiter,
    mode: Literal["text", "binary"] = "binary",
    show_progress: bool = True,
    enforce_extension: bool = True,
) -> BatchDownloadStats:
    logger = logging.getLogger(__name__ + "." + download_missing.__name__)
    urls_to_download, save_paths, skipped_existing = split_urls_by_existing_files(
        urls, 
        output_dir, 
        path_func,
        show_progress=show_progress
    )
    downloaded = 0
    failed = 0
    logger.info(
        "Starting batch download of URLs. %d total URLs, %d to download after skipping existing files.",
        len(urls),
        len(urls_to_download),
        extra={
            "total_urls": len(urls),
            "urls_to_download": len(urls_to_download),
            "skipped_existing": skipped_existing,
            "output_dir": str(output_dir),
        }
    )
    if show_progress:
        iterator = tqdm.tqdm(zip(urls_to_download, save_paths), total=len(urls_to_download), desc="Downloading")
    else:
        iterator = zip(urls_to_download, save_paths)
    for url, save_path in iterator:
        success = download_url(
            url=url,
            output_file=save_path,
            rate_limiter=rate_limiter,
            session=session,
            mode=mode,
        )
        if success:
            downloaded += 1
        else:
            failed += 1
    logger.info(
        "Completed batch download of URLs. %d downloaded, %d failed, %d skipped existing files.",
        downloaded,
        failed,
        skipped_existing,
        extra={
            "total_urls": len(urls),
            "urls_to_download": len(urls_to_download),
            "skipped_existing": skipped_existing,
            "downloaded": downloaded,
            "failed": failed,
            "output_dir": str(output_dir),
        }
    )
    return BatchDownloadStats(
        total=len(urls),
        queued=len(urls_to_download),
        skipped_existing=skipped_existing,
        downloaded=downloaded,
        failed=failed,
    )
