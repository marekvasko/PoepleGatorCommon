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
        "-s",
        "--output-skipped-csv",
        type=str,
        required=True,
        help="CSV file to save skipped entries.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force reprocessing of all entries. Otherwise diff between --input-dir and --output-csv will be used to determine which entries to process.",
    )
    parser.add_argument(
        "-w",
        "--num-workers",
        type=int,
        default=16,
        help="Number of worker processes to use for parsing HTML files.",
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
        output_skipped_csv=args.output_skipped_csv,
        show_progress=not args.hide_progress,
        diff_only=not args.force,
        num_workers=args.num_workers,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
