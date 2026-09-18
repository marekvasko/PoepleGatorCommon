"""Command-line interface for LLM-based PeopleGator NER extraction."""

from __future__ import annotations

import argparse

import dotenv

from people_gator_common.cli.helpers import configure_logging
from people_gator_common.pipelines.llm_ner import (
    DEFAULT_INPUT_PREFIX,
    DEFAULT_MIN_CHARACTERS_IN_CHUNK,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_PREFIX,
    DEFAULT_OVERLAP_LINES,
    run_llm_ner_pipeline,
)

dotenv.load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract NER between configured-storage PeopleGator prefixes.",
    )
    parser.add_argument(
        "-i",
        "--input-prefix",
        "--input-dir",
        dest="input_prefix",
        default=DEFAULT_INPUT_PREFIX,
        help=(
            "Storage-relative prefix containing "
            "<library>/<document>/<page>.json objects "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-prefix",
        "--output-dir",
        dest="output_prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help="Storage-relative output prefix for per-page NER JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenRouter model identifier (default: %(default)s).",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default="low",
        help="Model reasoning effort (default: %(default)s).",
    )
    parser.add_argument(
        "--min-characters-in-chunk",
        type=int,
        default=DEFAULT_MIN_CHARACTERS_IN_CHUNK,
        help="Target minimum chunk size before a new chunk (default: %(default)s).",
    )
    parser.add_argument(
        "--overlap-lines",
        type=int,
        default=DEFAULT_OVERLAP_LINES,
        help="Lines repeated between adjacent chunks (default: %(default)s).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        help="Process at most this many new pages (useful for smoke tests).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess pages whose output files already exist.",
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
    summary = run_llm_ner_pipeline(
        input_prefix=args.input_prefix,
        output_prefix=args.output_prefix,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        min_characters_in_chunk=args.min_characters_in_chunk,
        overlap_lines=args.overlap_lines,
        force=args.force,
        max_pages=args.max_pages,
    )
    print(
        f"LLM NER complete: {summary.processed_pages} processed, "
        f"{summary.skipped_pages} skipped"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
