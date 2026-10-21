"""Command-line interface for NER entity clustering and disambiguation."""

from __future__ import annotations

import argparse

import dotenv

from people_gator_common.cli.helpers import configure_logging
from people_gator_common.pipelines.ner_clusters import (
    DEFAULT_CLUSTER_THRESHOLD,
    DEFAULT_DOCUMENT_OUTPUT_PREFIX,
    DEFAULT_INPUT_PREFIX,
    DEFAULT_MENTION_WEIGHT,
    DEFAULT_MODEL,
    DEFAULT_NAME_WEIGHT,
    DEFAULT_OUTPUT_PREFIX,
    DEFAULT_PROGRESS_INTERVAL_SECONDS,
    run_ner_clusters_pipeline,
)

dotenv.load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster and disambiguate named entities from saved LLM NER "
            "page outputs between configured-storage prefixes."
        ),
    )
    parser.add_argument(
        "-i",
        "--input-prefix",
        "--input-dir",
        dest="input_prefix",
        default=DEFAULT_INPUT_PREFIX,
        help=(
            "Storage-relative prefix containing "
            "<library>/<document>/<page>.json LLM NER outputs "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-prefix",
        "--output-dir",
        dest="output_prefix",
        default=None,
        help=(
            "Storage-relative output prefix for one dump per selected scope unit "
            f"(default: {DEFAULT_OUTPUT_PREFIX} for page scope, "
            f"{DEFAULT_DOCUMENT_OUTPUT_PREFIX} for document scope)."
        ),
    )
    parser.add_argument(
        "--scope",
        choices=["page", "document"],
        default="page",
        help=(
            "Cluster within each page or across all available pages of each "
            "library/document. Write one dump per page or document with member "
            "page/chunk references (default: %(default)s)."
        ),
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
        "--threshold",
        type=float,
        default=DEFAULT_CLUSTER_THRESHOLD,
        help=(
            "Fuzzy name-clustering threshold (0-100). "
            "Higher = stricter matching (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--name-weight",
        type=float,
        default=DEFAULT_NAME_WEIGHT,
        help="Weight of canonical names in pairwise scoring (default: %(default)s).",
    )
    parser.add_argument(
        "--mention-weight",
        type=float,
        default=DEFAULT_MENTION_WEIGHT,
        help="Weight of mention names in pairwise scoring (default: %(default)s).",
    )
    parser.add_argument(
        "--no-skip-singletons",
        action="store_true",
        help="Do not skip LLM disambiguation for single-mention clusters.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        help="Process at most this many new pages (page scope only).",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        help="Process at most this many documents, including all their pages (document scope only).",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=DEFAULT_PROGRESS_INTERVAL_SECONDS,
        help="Seconds between progress/ETA logs (default: %(default)s).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate existing cluster dumps even when inputs are unchanged.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args()
    if args.scope == "document" and args.max_pages is not None:
        parser.error("--max-pages requires --scope page; use --max-documents")
    if args.scope == "page" and args.max_documents is not None:
        parser.error("--max-documents requires --scope document")
    if args.max_pages is not None and args.max_pages <= 0:
        parser.error("--max-pages must be positive")
    if args.max_documents is not None and args.max_documents <= 0:
        parser.error("--max-documents must be positive")
    return args


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    summary = run_ner_clusters_pipeline(
        input_prefix=args.input_prefix,
        output_prefix=args.output_prefix,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        threshold=args.threshold,
        mention_weight=args.mention_weight,
        name_weight=args.name_weight,
        force=args.force,
        max_pages=args.max_pages,
        scope=args.scope,
        max_documents=args.max_documents,
        progress_interval_seconds=args.progress_interval,
    )
    if args.scope == "document":
        print(
            f"NER clustering complete: {summary.processed_documents} documents processed, "
            f"{summary.skipped_documents} documents skipped; "
            f"{summary.processed_pages} pages processed, {summary.skipped_pages} pages skipped; "
            f"{summary.clusters} document clusters"
        )
    else:
        print(
            f"NER clustering complete: {summary.processed_pages} processed, "
            f"{summary.skipped_pages} skipped, {summary.clusters} clusters"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
