from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from people_gator_common.pipelines.obalkyknih import run_obalkyknih_html_download_pipeline
from people_gator_common.rate_limiter import RateLimiter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI for ObalkyKnih HTML download pipeline.")
    parser.add_argument(
        "-i",
        "--input-csv",
        type=Path,
        default=Path("output/unique_nl_cr_aut_id.csv"),
        help="CSV file containing NLCR AUT IDs.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save downloaded HTML files.",
    )
    parser.add_argument(
        "--auth-id-column",
        type=str,
        default="entity",
        help="Column name in --input-csv containing NLCR AUT IDs.",
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
        default=1.0,
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

    stats = run_obalkyknih_html_download_pipeline(
        input_csv=Path(args.input_csv),
        output_dir=Path(args.output_dir),
        rate_limiter=rate_limiter,
        auth_id_column=args.auth_id_column,
        show_progress=not bool(args.hide_progress),
    )

    logger = logging.getLogger(__name__)
    logger.info(
        "Obalkyknih pipeline finished. total=%d queued=%d skipped_existing=%d downloaded=%d failed=%d",
        getattr(stats, "total", 0),
        getattr(stats, "queued", 0),
        getattr(stats, "skipped_existing", 0),
        getattr(stats, "downloaded", 0),
        getattr(stats, "failed", 0),
        extra={
            "total": getattr(stats, "total", 0),
            "queued": getattr(stats, "queued", 0),
            "skipped_existing": getattr(stats, "skipped_existing", 0),
            "downloaded": getattr(stats, "downloaded", 0),
            "failed": getattr(stats, "failed", 0),
        },
    )
    return 1 if getattr(stats, "failed", 0) > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
