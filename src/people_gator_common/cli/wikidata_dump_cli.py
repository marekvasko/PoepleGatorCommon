"""CLI for extracting local Wikidata JSON dumps from YAML configuration."""

from __future__ import annotations

import argparse
import logging

from people_gator_common.cli.helpers import configure_logging
from people_gator_common.pipelines.wikidata_dump import (
    load_wikidata_dump_config,
    run_wikidata_dump_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream a local Wikidata JSON dump into configured CSV files."
    )
    parser.add_argument("--config", required=True, help="Path to YAML configuration.")
    parser.add_argument("--input-path", help="Override input_path from YAML.")
    parser.add_argument(
        "--output-directory", help="Override output_directory from YAML."
    )
    parser.add_argument("--workers", type=int, help="Override parser worker count.")
    parser.add_argument(
        "--resume-from-line",
        type=int,
        help=(
            "Append to existing outputs and resume at this inclusive, one-based "
            "physical dump line. Earlier lines are decompressed but not parsed."
        ),
    )
    parser.add_argument(
        "--force", action="store_true", help="Replace existing configured outputs."
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    config = load_wikidata_dump_config(args.config)
    updates: dict[str, object] = {}
    if args.input_path:
        updates["input_path"] = args.input_path
    if args.output_directory:
        updates["output_directory"] = args.output_directory
    if args.workers is not None:
        updates["workers"] = args.workers
    if args.resume_from_line is not None:
        updates["resume_from_line"] = args.resume_from_line
    if args.force:
        updates["overwrite"] = True
    if updates:
        config = type(config).model_validate(config.model_dump() | updates)
    stats = run_wikidata_dump_pipeline(config)
    logging.getLogger(__name__).info("Processed %d entities", stats.items_seen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
