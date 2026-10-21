"""CLI for linking disambiguated NER clusters to API entities."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os

import dotenv

from people_gator_common.cli.helpers import configure_logging
from people_gator_common.pipelines.ner_clusters import DEFAULT_MODEL
from people_gator_common.pipelines.ner_entity_linking import (
    DEFAULT_API_BASE_URL,
    DEFAULT_DOCUMENT_INPUT_PREFIX,
    DEFAULT_DOCUMENT_OUTPUT_PREFIX,
    DEFAULT_INPUT_PREFIX,
    DEFAULT_OUTPUT_PREFIX,
    DEFAULT_PROGRESS_INTERVAL_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    EntityLinkingConfig,
    rollback_latest_entity_linking_run,
    run_ner_entity_linking_pipeline,
)

dotenv.load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search People Gator for each disambiguated NER cluster, let an "
            "LLM choose an existing entity or a new one, apply the decision, "
            "and upload source text mentions with a durable audit trail."
        )
    )
    parser.add_argument(
        "-i",
        "--input-prefix",
        "--input-dir",
        dest="input_prefix",
        default=None,
        help=(
            f"Cluster dump input prefix (default: {DEFAULT_INPUT_PREFIX} for page scope, "
            f"{DEFAULT_DOCUMENT_INPUT_PREFIX} for document scope)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-prefix",
        "--output-dir",
        dest="output_prefix",
        default=None,
        help=(
            f"Link journal output prefix (default: {DEFAULT_OUTPUT_PREFIX} for page scope, "
            f"{DEFAULT_DOCUMENT_OUTPUT_PREFIX} for document scope)."
        ),
    )
    parser.add_argument(
        "--scope",
        choices=["page", "document"],
        default="page",
        help="Read one cluster dump and save one link journal per page or document (default: %(default)s).",
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
        "--refresh-token",
        default=os.getenv("PEOPLE_GATOR_API_REFRESH_TOKEN"),
        help="Refresh token paired with the supplied bearer token.",
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
        help="Process at most this many new pages (page scope only).",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        help="Process at most this many documents, including every member page (document scope only).",
    )
    parser.add_argument(
        "--parallel-entities",
        "--parallel-pages",
        "--parallel-requests",
        dest="parallel_pages",
        type=int,
        default=4,
        help=(
            "Maximum number of entity searches and LLM decisions in flight "
            "(default: %(default)s). The older --parallel-pages and "
            "--parallel-requests aliases are also accepted."
        ),
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
        help="Reprocess existing page/document journals; live mutation journals must be rolled back first.",
    )
    parser.add_argument(
        "--decision-only",
        action="store_true",
        help="Save decisions without changing entities or uploading mentions.",
    )
    parser.add_argument(
        "--rollback-latest",
        action="store_true",
        help="Reverse the latest page/document journal instead of processing inputs.",
    )
    parser.add_argument(
        "--force-rollback",
        action="store_true",
        help="Rollback even if a recorded entity has since changed.",
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


async def _run(args: argparse.Namespace) -> int:
    config = EntityLinkingConfig(
        base_url=args.api_base_url,
        bearer_token=args.bearer_token,
        refresh_token=args.refresh_token,
        username=args.username,
        password=args.password,
        timeout_seconds=args.request_timeout_seconds,
    )
    if args.rollback_latest:
        result = await rollback_latest_entity_linking_run(
            config=config,
            output_prefix=args.output_prefix
            or (
                DEFAULT_DOCUMENT_OUTPUT_PREFIX
                if args.scope == "document"
                else DEFAULT_OUTPUT_PREFIX
            ),
            force=args.force_rollback,
        )
        print(
            f"Rolled back {result.rolled_back_mutations} mutations from "
            f"{result.audit_path}"
        )
        return 0

    summary = await run_ner_entity_linking_pipeline(
        config=config,
        input_prefix=args.input_prefix,
        output_prefix=args.output_prefix,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        force=args.force,
        max_pages=args.max_pages,
        scope=args.scope,
        max_documents=args.max_documents,
        progress_interval_seconds=args.progress_interval,
        mode="decision-only" if args.decision_only else "apply",
        parallel_pages=args.parallel_pages,
    )
    documents = (
        f"{summary.processed_documents} documents processed, {summary.skipped_documents} documents skipped; "
        if args.scope == "document"
        else ""
    )
    print(
        f"NER entity linking complete: {documents}{summary.processed_pages} pages processed, "
        f"{summary.skipped_pages} skipped; {summary.merge_decisions} merge and "
        f"{summary.create_decisions} create decisions saved for "
        f"{summary.source_entities} source entities; "
        f"{summary.entity_mutations} entity mutations and "
        f"{summary.uploaded_mentions} mentions applied"
    )
    return 0


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    logging.getLogger("httpx").disabled = True
    logging.getLogger("httpcore").disabled = True
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
