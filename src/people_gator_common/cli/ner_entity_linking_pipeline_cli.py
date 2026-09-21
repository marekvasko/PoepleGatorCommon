"""CLI for linking disambiguated NER clusters to API entities."""

from __future__ import annotations

import argparse
import asyncio
import os

import dotenv

from people_gator_common.cli.helpers import configure_logging
from people_gator_common.pipelines.ner_clusters import DEFAULT_MODEL
from people_gator_common.pipelines.ner_entity_linking import (
    DEFAULT_API_BASE_URL,
    DEFAULT_INPUT_PREFIX,
    DEFAULT_OUTPUT_PREFIX,
    DEFAULT_PROGRESS_INTERVAL_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    EntityLinkingConfig,
    run_ner_entity_linking_pipeline,
)

dotenv.load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search People Gator for each disambiguated NER cluster, let an "
            "LLM propose an existing entity or a new one, and save the decision "
            "without changing API entities."
        )
    )
    parser.add_argument(
        "-i",
        "--input-prefix",
        "--input-dir",
        dest="input_prefix",
        default=DEFAULT_INPUT_PREFIX,
        help="Storage prefix containing per-page NER cluster JSON (default: %(default)s).",
    )
    parser.add_argument(
        "-o",
        "--output-prefix",
        "--output-dir",
        dest="output_prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help="Storage prefix for per-page entity-link JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--api-base-url",
        default=DEFAULT_API_BASE_URL,
        help="People Gator API base URL (default: %(default)s).",
    )
    parser.add_argument(
        "--bearer-token",
        default=os.getenv("PEOPLE_GATOR_API_BEARER_TOKEN"),
        help="Existing People Gator bearer token.",
    )
    parser.add_argument(
        "--username",
        default=os.getenv("PEOPLE_GATOR_API_USERNAME"),
        help="Username for the OAuth2 password flow when no token is supplied.",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("PEOPLE_GATOR_API_PASSWORD"),
        help="Password for the OAuth2 password flow when no token is supplied.",
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
        "--max-pages",
        type=int,
        help="Process at most this many new pages.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help="API timeout in seconds (default: %(default)s).",
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
        help="Reprocess pages whose entity-link output already exists.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> int:
    summary = await run_ner_entity_linking_pipeline(
        config=EntityLinkingConfig(
            base_url=args.api_base_url,
            bearer_token=args.bearer_token,
            username=args.username,
            password=args.password,
            timeout_seconds=args.request_timeout_seconds,
        ),
        input_prefix=args.input_prefix,
        output_prefix=args.output_prefix,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        force=args.force,
        max_pages=args.max_pages,
        progress_interval_seconds=args.progress_interval,
    )
    print(
        f"NER entity linking complete: {summary.processed_pages} processed, "
        f"{summary.skipped_pages} skipped; {summary.merge_decisions} merge and "
        f"{summary.create_decisions} create decisions saved for "
        f"{summary.source_entities} source entities"
    )
    return 0


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
