import argparse
import logging
import sys
from pathlib import Path

from people_gator_common.pipelines.wikicommons import run_wikicommons_images_download_pipeline
from people_gator_common.rate_limiter import RateLimiter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI for Wikicommons image download pipeline.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("output/people_all_P18.csv"),
        help="CSV file containing image URLs.",
    )
    parser.add_argument(
        "--url-column",
        type=str,
        default="value",
        help="Column name in --input-csv containing image URLs.",
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, required=True, help="Directory to save output files."
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


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
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
    stats = run_wikicommons_images_download_pipeline(
        input_csv=Path(args.input_csv),
        output_dir=output_dir,
        rate_limiter=rate_limiter,
        url_column=args.url_column,
        show_progress=not bool(args.hide_progress),
    )
    logger = logging.getLogger(__name__)
    logger.info(
        "Wikicommons pipeline finished. total=%d queued=%d skipped_existing=%d downloaded=%d failed=%d",
        stats.total,
        stats.queued,
        stats.skipped_existing,
        stats.downloaded,
        stats.failed,
    )
    return 1 if stats.failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
