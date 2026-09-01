from __future__ import annotations

import argparse
import asyncio

from people_gator_common.cli.helpers import configure_logging
from people_gator_common.pipelines.authority_import import ApiConnectionConfig
from people_gator_common.pipelines.authority_import import run_authority_import


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import authority records from Wikidata/Wikimedia and NLCR outputs into the People Gator API."
        )
    )
    parser.add_argument(
        "--wikidata-dir",
        type=str,
        default=None,
        help=(
            "Directory/prefix with Wikidata outputs: <prefix>.csv, <prefix>_labels.csv, "
            "<prefix>_descriptions.csv, <prefix>_alt_labels.csv, optional <prefix>_P18.csv."
        ),
    )
    parser.add_argument(
        "--wikidata-prefix",
        type=str,
        default="people_of_all",
        help="Filename prefix used in Wikidata CSV outputs.",
    )
    parser.add_argument(
        "--nlcr-dir",
        type=str,
        default=None,
        help=(
            "Directory/prefix with NLCR outputs: unique_control_numbers.csv, "
            "nlcr_aut__person_names_reparsed.csv, nlcr_aut__also_known_as_reparsed.csv, "
            "nlcr_aut__descriptions.csv."
        ),
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap for number of imported records.",
    )
    parser.add_argument(
        "--api-base-url",
        type=str,
        default="https://dev.people-gator.capturemyhand.com",
        help="Base URL of the People Gator API.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional API key or bearer token for the People Gator API.",
    )
    parser.add_argument(
        "--embedding-service-url",
        type=str,
        default=None,
        help="Optional embedding service endpoint that accepts a JSON body with a text field.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=30.0,
        help="HTTP timeout for API calls.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> int:
    await run_authority_import(
        wikidata_dir=args.wikidata_dir,
        wikidata_prefix=args.wikidata_prefix,
        nlcr_dir=args.nlcr_dir,
        max_records=args.max_records,
        weaviate=ApiConnectionConfig(
            base_url=args.api_base_url,
            api_key=args.api_key,
            embedding_service_url=args.embedding_service_url,
            timeout_seconds=args.request_timeout_seconds,
        ),
    )
    return 0


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
