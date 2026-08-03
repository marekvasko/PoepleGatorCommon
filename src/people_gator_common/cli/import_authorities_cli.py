from __future__ import annotations

import argparse
import asyncio

from people_gator_common.cli.helpers import configure_logging
from people_gator_common.pipelines.authority_import import WeaviateConnectionConfig
from people_gator_common.pipelines.authority_import import run_authority_import


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import authority records from Wikidata/Wikimedia and NLCR outputs into Weaviate."
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
        "--weaviate-http-host",
        type=str,
        default="localhost",
        help="Weaviate HTTP host.",
    )
    parser.add_argument(
        "--weaviate-http-port",
        type=int,
        default=8080,
        help="Weaviate HTTP port.",
    )
    parser.add_argument(
        "--weaviate-http-secure",
        action="store_true",
        help="Use HTTPS for Weaviate HTTP connection.",
    )
    parser.add_argument(
        "--weaviate-grpc-host",
        type=str,
        default="localhost",
        help="Weaviate gRPC host.",
    )
    parser.add_argument(
        "--weaviate-grpc-port",
        type=int,
        default=50051,
        help="Weaviate gRPC port.",
    )
    parser.add_argument(
        "--weaviate-grpc-secure",
        action="store_true",
        help="Use TLS for Weaviate gRPC connection.",
    )
    parser.add_argument(
        "--weaviate-api-key",
        type=str,
        default=None,
        help="Optional Weaviate API key.",
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
        weaviate=WeaviateConnectionConfig(
            http_host=args.weaviate_http_host,
            http_port=args.weaviate_http_port,
            http_secure=args.weaviate_http_secure,
            grpc_host=args.weaviate_grpc_host,
            grpc_port=args.weaviate_grpc_port,
            grpc_secure=args.weaviate_grpc_secure,
            api_key=args.weaviate_api_key,
        ),
    )
    return 0


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
