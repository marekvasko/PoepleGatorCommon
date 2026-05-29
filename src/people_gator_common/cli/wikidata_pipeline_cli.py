from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from people_gator_common.pipelines.wikidata import run_wikidata_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI for Wikidata DAG pipelines.")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="output/wikidata",
        type=str,
        help="Directory to save output files.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    parser.add_argument(
        "--receive-batch-size",
        type=int,
        default=40_000,
        help="Batch size for SPARQL collector queries.",
    )
    parser.add_argument(
        "--send-batch-size",
        type=int,
        default=10_000,
        help="Batch size for SPARQL enrichment queries.",
    )
    parser.add_argument(
        "--rate-limit-interval",
        type=float,
        default=5.0,
        help="Minimum interval between SPARQL requests in seconds.",
    )
    parser.add_argument(
        "--rate-limit-retries",
        type=int,
        default=5,
        help="Maximum number of retries for SPARQL requests.",
    )
    parser.add_argument(
        "--rate-limit-backoff",
        type=float,
        default=15.0,
        help="Backoff time in seconds between retries for SPARQL requests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    run_wikidata_pipeline(
        output_dir=str(args.output_dir),
        receive_batch_size=int(args.receive_batch_size),
        send_batch_size=int(args.send_batch_size),
        rate_limit_interval=float(args.rate_limit_interval),
        rate_limit_retries=int(args.rate_limit_retries),
        rate_limit_backoff=float(args.rate_limit_backoff),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
