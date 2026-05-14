from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from people_gator_common.pipelines.obalkyknih import run_obalkyknih_html_parse_pipeline
from people_gator_common.rate_limiter import RateLimiter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI for ObalkyKnih HTML download pipeline.")
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default="output/unique_nl_cr_aut_id.csv",
        help="CSV file containing NLCR AUT IDs.",
    )
    parser.add_argument(
        "-o",
        "--output-csv",
        type=str,
        required=True,
        help="CSV file to save extracted preview image URLs.",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    df = run_obalkyknih_html_parse_pipeline(
        html_dir=args.input_dir,
        output_csv=args.output_csv,
        show_progress=not args.hide_progress,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
