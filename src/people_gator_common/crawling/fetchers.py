from __future__ import annotations
from collections.abc import Callable, Sequence
import os
from typing import Literal
from requests import Session
from people_gator_common.rate_limiter import RateLimiter, RateLimiterTooManyRequestsError
from people_gator_common.errors import is_retryable_error
from people_gator_common.filesytem import exists, ensure_parent_dir, glob, write_text, write_bytes
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

def strip_extensions(str_path: str) -> str:
    return os.path.splitext(str_path)[0]

def split_urls_by_existing_files(
    urls: Sequence[str], 
    output_dir: str,
    path_func: PathFunc,
    *,
    show_progress: bool = True,
    ignore_extension: bool = True,
) -> tuple[list[str], int]:
    logger = logging.getLogger(__name__ + "." + split_urls_by_existing_files.__name__)
    
    urls_to_download: list[str] = []
    glob_pattern = os.path.join(str(output_dir), "**", "*")
    globbed_files = glob(glob_pattern)
    logger.info(
        "Globbed existing files in output directory: %s with pattern: %s. Found %d existing files. Sample of first 2 existing files: %s",
        output_dir, 
        glob_pattern, 
        len(globbed_files),
        globbed_files[:2],
        extra={
            "output_dir": str(output_dir),
            "glob_pattern": glob_pattern,
            "existing_files_count": len(globbed_files),
        }
    )
    if ignore_extension:
        existing_files = {strip_extensions(path) for path in globbed_files}
    else:
        existing_files = set(globbed_files)
    save_paths = [path_func(url, output_dir) for url in urls]
    logger.info(
        "Derived %d save paths for URLs using path function. Sample of first 2 URLs and their save paths: %s",
        len(save_paths),
        list(zip(urls, save_paths))[:2],
        extra={
            "sample_urls_and_paths": list(zip(urls, save_paths))[:2],
        }
    )
    if ignore_extension:
        transformed_save_paths = [strip_extensions(str(path)) if path is not None else None for path in save_paths]
    else:
        transformed_save_paths = [str(path) if path is not None else None for path in save_paths]
    
    if show_progress:
        iterator = tqdm.tqdm(zip(urls, transformed_save_paths), total=len(urls), desc="Checking existing files")
    else:
        iterator = zip(urls, transformed_save_paths)
    
    for url, save_path in iterator:
        if save_path is None:
            logger.warning("No valid save path for url %s; skipping existence check and download", url, extra={"url": url})
            urls_to_download.append(url)
            continue
        if save_path in existing_files:
            logger.info("File already exists for url %s at path %s; skipping download", url, save_path, extra={"url": url, "save_path": save_path})
        else:
            urls_to_download.append(url)
    
    skipped_existing = len(urls) - len(urls_to_download)
    logger.info(
        "Completed checking existing files. %d URLs to download, %d skipped because file already exists.",
        len(urls_to_download), skipped_existing,
        extra={
            "total_urls": len(urls),
            "urls_to_download": len(urls_to_download),
            "skipped_existing": skipped_existing,
        }
    )
    
    return urls_to_download, skipped_existing

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

def download_url(
    url: str,
    output_dir: str,
    *,
    path_func: PathFunc,
    session: Session,
    rate_limiter: RateLimiter,
    mode: Literal["text", "binary"] = "binary",
    enforce_extension: bool = True,
) -> bool:
    logger = logging.getLogger(__name__ + "." + download_url.__name__)
    save_path = path_func(url, output_dir)
    if save_path is None:
        logger.warning("No valid save path for url %s; skipping download", url, extra={"url": url})
        return False

    ensure_parent_dir(save_path)
    with rate_limiter as rl:
        while True:
            try:
                rl.request()
                logger.info(
                    "Downloading url %s with mode %s",
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
                if enforce_extension and "Content-Type" in response.headers:
                    content_type = response.headers["Content-Type"].split(";")[0].strip()
                    save_path = enforce_path_extension_from_mimetype(save_path, content_type)
                if mode == "text":
                    write_text(save_path, response.text)
                elif mode == "binary":
                    write_bytes(save_path, response.content)
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
                    "Too many requests error for url %s: %s", 
                    url, error, 
                    extra={"url": url}
                )
                return False
            except Exception as error:
                logger.warning(
                    "Error fetching url %s: %s", 
                    url, error, 
                    extra={"url": url}
                )
                if is_retryable_error(error):
                    logger.info(
                        "Retrying after error for url %s", 
                        url, 
                        extra={"url": url}
                    )
                    continue
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
    urls_to_download, skipped_existing = split_urls_by_existing_files(
        urls, 
        output_dir, 
        path_func, 
        ignore_extension=enforce_extension, 
        show_progress=show_progress
    )
    downloaded = 0
    failed = 0

    iterator: Sequence[str] | tqdm.tqdm[str]
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
            enforce_extension=enforce_extension,
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
