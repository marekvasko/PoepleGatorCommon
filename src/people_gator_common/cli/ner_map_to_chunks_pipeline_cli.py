"""CLI for mapping NER entities to character spans in source documents.

Enriches disambiguated entity clusters with fuzzy-matched span positions,
scores, and extracted text snippets from the original document corpus.
"""

from __future__ import annotations

import argparse

import dotenv

from people_gator_common.cli.helpers import configure_logging
from people_gator_common.pipelines.ner_map_to_chunks import run_ner_map_to_chunks_pipeline

dotenv.load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map disambiguated NER entities to character spans in source documents.",
    )
    parser.add_argument(
        "-d", "--documents-dir",
        type=str,
        required=True,
        help="Directory containing per-document JSONL files with 'id' and 'text' fields.",
    )
    parser.add_argument(
        "-l", "--linked-clusters-dir",
        type=str,
        required=True,
        help="Directory containing disambiguated entity cluster JSON files.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="ner_postprocess_output/llmlinked_ner_clusters_with_spans",
        help="Output directory for enriched cluster JSON files (default: %(default)s).",
    )
    parser.add_argument(
        "--max-l-dist",
        type=int,
        default=5,
        help="Maximum Levenshtein distance for fuzzy substring matching (default: 5).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=80.0,
        help="Minimum span confidence score 0–100 to keep a match (default: 80.0).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    run_ner_map_to_chunks_pipeline(
        documents_dir=args.documents_dir,
        linked_clusters_dir=args.linked_clusters_dir,
        output_dir=args.output_dir,
        max_l_dist=args.max_l_dist,
        score_threshold=args.score_threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
