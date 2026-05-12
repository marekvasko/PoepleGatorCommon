import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from requests import Session
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from people_gator_common.images.download import (
    download_image,
    get_save_path,
)
from people_gator_common.rate_limiter import RateLimiter
from people_gator_common.headers import CLIENT_HEADERS

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI for Wikidata DAG pipelines.")
    parser.add_argument(
        "-o", "--output-dir", type=Path, required=True, help="Directory to save output files."
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
        default=5.0,
        help="Minimum interval between SPARQL requests in seconds (default: 5.0).",
    )
    parser.add_argument(
        "--rate-limit-retries",
        type=int,
        default=5,
        help="Maximum number of retries for SPARQL requests (default: 5).",
    )
    parser.add_argument(
        "--rate-limit-backoff",
        type=float,
        default=15.0,
        help="Backoff time in seconds between retries for SPARQL requests (default: 15.0).",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    output_dir = args.output_dir
    log_level = getattr(logging, args.log_level)
    rate_limit_interval = float(args.rate_limit_interval)
    rate_limit_retries = int(args.rate_limit_retries)
    rate_limit_backoff = float(args.rate_limit_backoff)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    rate_limiter = RateLimiter(
        max_retries=rate_limit_retries,
        retry_backoff_seconds=rate_limit_backoff,
        min_request_interval=rate_limit_interval,
    )

    with logging_redirect_tqdm(), Session() as session:
        session.headers.update(CLIENT_HEADERS)
        
        retrieves_images_df = pd.read_csv("output/people_all_P18.csv")
        save_paths = retrieves_images_df["value"].apply(lambda url: get_save_path(str(url), save_root=output_dir)) # type: ignore
        save_path_exists = save_paths.apply(lambda path: path is not None and path.exists())
        logger = logging.getLogger(__name__)
        logger.info("Total images to process: %d", retrieves_images_df.shape[0])
        logger.info("Images already downloaded: %d", save_path_exists.sum())
        retrieves_images_df = retrieves_images_df[~save_path_exists].reset_index(drop=True)

        for _, row in tqdm.tqdm(retrieves_images_df.iterrows(), total=retrieves_images_df.shape[0]):
            image_url = row["value"]
            download_image(
                image_url,
                output_dir=output_dir,
                rate_limiter=rate_limiter,
                session=session,
            )

if __name__ == "__main__":
    main()
