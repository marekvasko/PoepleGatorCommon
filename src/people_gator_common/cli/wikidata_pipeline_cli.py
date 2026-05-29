from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from people_gator_common.cli.helpers import add_rate_limiter_args, configure_logging, rate_limiter_from_args
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
        help="Logging level.",
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
    add_rate_limiter_args(parser, default_interval=5.0, default_retries=5, default_backoff=15.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    run_wikidata_pipeline(
        output_dir=str(args.output_dir),
        receive_batch_size=int(args.receive_batch_size),
        send_batch_size=int(args.send_batch_size),
        rate_limiter=rate_limiter_from_args(args),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
