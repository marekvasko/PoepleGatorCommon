from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from requests import Session
from tqdm.contrib.logging import logging_redirect_tqdm

from people_gator_common.crawling.fetchers import download_missing
from os.path import join
from people_gator_common.filesytem import open_file
from people_gator_common.headers import CLIENT_HEADERS
from people_gator_common.rate_limiter import RateLimiter

def obalkyknih_image_path_func(url: str, output_dir: str) -> str | None:
    logger = logging.getLogger(__name__ + "." + obalkyknih_image_path_func.__name__)
    if not url.startswith("http"):
        logger.warning("Skipping non-http URL: %s", url)
        return None
    parsed_url = urlparse(url)
    netloc = parsed_url.netloc.replace(":", "_")
    split_path = parsed_url.path.lstrip("/").split("/")
    path = join(*split_path[:-1]) + "_" + split_path[-1]
    filename = join(output_dir, netloc, f"{path}.png")
    return filename

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI for ObalkyKnih HTML download pipeline.")
    parser.add_argument(
        "-i",
        "--input-csv",
        type=str,
        default="obalkyknih/previewimage.csv",
        help="CSV file containing URLs of preview images.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="images",
        help="Directory to save downloaded HTML files.",
    )
    parser.add_argument(
        "--image-column",
        type=str,
        default="value",
        help="Column name in --input-csv containing URLs of preview images.",
    )
    parser.add_argument(
        "--hide-progress",
        action="store_true",
        help="Disable tqdm progress bar output.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    parser.add_argument(
        "--rate-limit-interval",
        type=float,
        default=0.5,
        help="Minimum interval between requests in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--rate-limit-retries",
        type=int,
        default=3,
        help="Maximum number of retries for requests (default: 3).",
    )
    parser.add_argument(
        "--rate-limit-backoff",
        type=float,
        default=5.0,
        help="Backoff time in seconds between retries (default: 5.0).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    rate_limiter = RateLimiter(
        min_request_interval=float(args.rate_limit_interval),
        max_retries=int(args.rate_limit_retries),
        retry_backoff_seconds=float(args.rate_limit_backoff),
    )


    logger = logging.getLogger(__name__)
    with open_file(args.input_csv, "r") as f:
        urls_df = pd.read_csv(f)
    
    if args.image_column not in urls_df.columns:
        logger.error("CSV file %s must contain '%s' column", args.input_csv, args.image_column)
        return 1
    urls: list[str] = [ url for url in urls_df[args.image_column].dropna().astype(str).str.strip() if url  != "" ]
    with Session() as session, logging_redirect_tqdm():
        session.headers.update(CLIENT_HEADERS)
        stats = download_missing(
            urls=urls,
            output_dir=args.output_dir,
            path_func=obalkyknih_image_path_func,
            rate_limiter=rate_limiter,
            mode="binary",
            show_progress=not args.hide_progress,
            session=session,
        )
    logger.info("Download completed: %d total, %d queued, %d skipped existing, %d downloaded, %d failed",
        stats.queued,
        stats.queued,
        stats.skipped_existing,
        stats.downloaded,
        stats.failed,
        extra={
            "total_urls": stats.total,
            "queued": stats.queued,
            "skipped_existing": stats.skipped_existing,
            "downloaded": stats.downloaded,
            "failed": stats.failed,
            "output_dir": args.output_dir,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
