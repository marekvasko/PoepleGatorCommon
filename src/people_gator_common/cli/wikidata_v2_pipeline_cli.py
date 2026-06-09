"""CLI for the Wikidata v2 manifest-based collection pipeline."""

from __future__ import annotations

import argparse
import logging
import sys

from people_gator_common.cli.helpers import add_rate_limiter_args, configure_logging, rate_limiter_from_args
from people_gator_common.pipelines.wikidata_v2 import run_wikidata_v2_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLI for the Wikidata v2 manifest-based collection pipeline.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="wikidata_v2",
        type=str,
        help="Directory to save output files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force overwrite of existing manifests and partial results.",
    )
    parser.add_argument(
        "--tmp-dir",
        default="tmp/wikidata",
        type=str,
        help="Directory to save temporary manifest and partial result files.",
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
        default=300_000,
        help="Batch size for SPARQL collector queries (entity discovery pagination).",
    )
    parser.add_argument(
        "--send-batch-size",
        type=int,
        default=30_000,
        help="Batch size for SPARQL enrichment queries (entity batch pagination).",
    )
    add_rate_limiter_args(parser, default_interval=1.0, default_retries=5, default_backoff=15.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    run_wikidata_v2_pipeline(
        output_dir=str(args.output_dir),
        tmp_dir=str(args.tmp_dir),
        receive_batch_size=int(args.receive_batch_size),
        send_batch_size=int(args.send_batch_size),
        force=bool(args.force),
        rate_limiter=rate_limiter_from_args(args),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
