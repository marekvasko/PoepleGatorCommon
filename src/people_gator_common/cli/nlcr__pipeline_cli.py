from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from people_gator_common.pipelines.nlcr import run_obalkyknih_html_pipeline
from people_gator_common.rate_limiter import RateLimiter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NLCR -> ObalkyKnih HTML download pipeline.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("output/unique_nl_cr_aut_id.csv"),
        help="CSV file containing NLCR AUT IDs.",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default="entity",
        help="Column name in --input-csv containing NLCR AUT IDs.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("output/obalkyknih_html"),
        help="Directory where downloaded HTML files will be saved.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of IDs to process.",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Do not deduplicate IDs from the input CSV.",
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
        help="Logging level.",
    )
    parser.add_argument(
        "--rate-limit-interval",
        type=float,
        default=5.0,
        help="Minimum interval between requests in seconds.",
    )
    parser.add_argument(
        "--rate-limit-retries",
        type=int,
        default=3,
        help="Maximum number of retries for request throttling and retryable failures.",
    )
    parser.add_argument(
        "--rate-limit-backoff",
        type=float,
        default=5.0,
        help="Retry backoff in seconds.",
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
    stats = run_obalkyknih_html_pipeline(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        id_column=args.id_column,
        deduplicate_ids=not bool(args.keep_duplicates),
        limit=args.limit,
        rate_limiter=rate_limiter,
        show_progress=not bool(args.hide_progress),
    )

    logger = logging.getLogger(__name__)
    logger.info(
        "NLCR pipeline finished. total=%d downloaded=%d skipped_existing=%d failed=%d",
        stats.total,
        stats.downloaded,
        stats.skipped_existing,
        stats.failed,
    )
    return 1 if stats.failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
