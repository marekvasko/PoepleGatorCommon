from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import tqdm
from requests import Session
from tqdm.contrib.logging import logging_redirect_tqdm

from people_gator_common.headers import CLIENT_HEADERS
from people_gator_common.images.download import download_image, get_save_path
from people_gator_common.rate_limiter import RateLimiter


@dataclass(frozen=True, slots=True)
class WikiCommonsDownloadStats:
    total: int
    queued: int
    skipped_existing: int
    downloaded: int
    failed: int


def _extract_image_urls_from_csv(image_csv: Path, *, url_column: str) -> list[str]:
    images_df = pd.read_csv(image_csv)
    if url_column not in images_df.columns:
        raise ValueError(f"CSV file {image_csv} must contain '{url_column}' column")
    urls = images_df[url_column].dropna().astype(str).str.strip()
    urls = urls[urls != ""]
    return urls.tolist()


def _split_urls_by_existing_files(urls: list[str], output_dir: Path) -> tuple[list[str], int]:
    urls_to_download: list[str] = []
    skipped_existing = 0
    for url in urls:
        save_path = get_save_path(url, save_root=output_dir)
        if save_path is not None and save_path.exists():
            skipped_existing += 1
            continue
        urls_to_download.append(url)
    return urls_to_download, skipped_existing


def run_wikicommons_images_download_pipeline(
    input_csv: Path,
    output_dir: Path,
    *,
    rate_limiter: RateLimiter,
    url_column: str = "value",
    show_progress: bool = True,
) -> WikiCommonsDownloadStats:
    image_urls = _extract_image_urls_from_csv(input_csv, url_column=url_column)
    urls_to_download, skipped_existing = _split_urls_by_existing_files(image_urls, output_dir)

    progress_context = logging_redirect_tqdm() if show_progress else nullcontext()
    with progress_context, Session() as session:
        session.headers.update(CLIENT_HEADERS)
        downloaded = 0
        failed = 0
        iterator = tqdm.tqdm(urls_to_download, total=len(urls_to_download)) if show_progress else urls_to_download
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

    return WikiCommonsDownloadStats(
        total=len(image_urls),
        queued=len(urls_to_download),
        skipped_existing=skipped_existing,
        downloaded=downloaded,
        failed=failed,
    )
