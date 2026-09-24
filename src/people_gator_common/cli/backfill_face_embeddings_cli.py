from __future__ import annotations

import argparse
import asyncio
import os

from people_gator_common.cli.helpers import configure_logging
from people_gator_common.pipelines.face_import import (
    FaceImportConfig,
    run_face_embedding_backfill,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach a precomputed embedding artifact to existing API faces."
    )
    parser.add_argument("embeddings", help="Path to embeddings.npy")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument(
        "--api-base-url",
        default="https://dev.people-gator.capturemyhand.com",
    )
    parser.add_argument(
        "--bearer-token", default=os.getenv("PEOPLE_GATOR_API_BEARER_TOKEN")
    )
    parser.add_argument("--username", default=os.getenv("PEOPLE_GATOR_API_USERNAME"))
    parser.add_argument("--password", default=os.getenv("PEOPLE_GATOR_API_PASSWORD"))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--request-timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> int:
    stats = await run_face_embedding_backfill(
        args.embeddings,
        manifest_path=args.manifest,
        metadata_path=args.metadata,
        config=FaceImportConfig(
            base_url=args.api_base_url,
            bearer_token=args.bearer_token,
            username=args.username,
            password=args.password,
            batch_size=args.batch_size,
            timeout_seconds=args.request_timeout_seconds,
        ),
    )
    print(
        f"Attached {stats.embeddings_updated} embeddings to {stats.faces_found} "
        f"faces in {stats.batches} batches."
    )
    return 0


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
