"""CLI for the NER post-processing pipeline.

Resolves named entities extracted from historical OCR documents into unified
real-world referents with common names, merged descriptions, and source-entity
mappings.

The OpenRouter API key is read from the ``NER_POSTPROCESS_API_KEY``
environment variable (or ``OPENROUTER_API_KEY`` as fallback).
"""

from __future__ import annotations

import argparse
import os

import dotenv

from people_gator_common.cli.helpers import configure_logging
from people_gator_common.pipelines.ner_postprocess import run_ner_postprocess_pipeline

dotenv.load_dotenv()


def _resolve_api_key() -> str:
    """Resolve the API key from environment variables."""
    key = os.environ.get("NER_POSTPROCESS_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise EnvironmentError(
            "No API key found. Set NER_POSTPROCESS_API_KEY or OPENROUTER_API_KEY "
            "in your environment or .env file."
        )
    return key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-process named entities: normalize → fuzzy cluster → LLM disambiguate.",
    )
    parser.add_argument(
        "-i", "--input-dir",
        type=str,
        required=True,
        help="Directory containing per-document NER JSONL extraction files.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="ner_postprocess_output",
        help="Root output directory for all pipeline artifacts (default: ner_postprocess_output).",
    )
    parser.add_argument(
        "--entity-type",
        type=str,
        default="being",
        help="NER entity type to process (default: being).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek/deepseek-v4-pro",
        help="LLM model identifier for the disambiguation step (default: deepseek/deepseek-v4-pro).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://openrouter.ai/api/v1",
        help="OpenRouter-compatible API base URL.",
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=92.0,
        help="Fuzzy clustering threshold (0-100). Higher = stricter matching (default: 92.0).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel workers for fuzzy scoring (default: cpu_count-1).",
    )
    parser.add_argument(
        "--no-skip-singletons",
        action="store_true",
        default=False,
        help="Do NOT skip LLM calls for single-entity clusters.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-run all steps even if intermediate outputs already exist.",
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

    api_key = _resolve_api_key()

    from multiprocessing import cpu_count

    n_jobs = args.n_jobs if args.n_jobs is not None else max(1, cpu_count() - 1)

    run_ner_postprocess_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        api_key=api_key,
        model=args.model,
        base_url=args.base_url,
        entity_type=args.entity_type,
        fuzzy_threshold=args.fuzzy_threshold,
        skip_singletons=not args.no_skip_singletons,
        n_jobs=n_jobs,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
