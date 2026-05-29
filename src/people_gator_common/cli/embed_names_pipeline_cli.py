from __future__ import annotations

import argparse

import dotenv

from people_gator_common.cli.helpers import configure_logging
from people_gator_common.pipelines.embed_names import run_embed_names_pipeline

dotenv.load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI for embedding person names via the sentence-transformer service.")
    parser.add_argument("--input-path", type=str, default="embeddings/names.csv", help="Path to the CSV file containing a 'name' column.")
    parser.add_argument("--output-path", type=str, default="embeddings/names_embeddings.npy", help="Path to write the resulting embeddings as a NumPy .npy file.")
    parser.add_argument("--service-url", type=str, default="https://dev.people-gator-services.capturemyhand.com/api/sentence-transformer/predict", help="URL of the sentence-transformer prediction endpoint.")
    parser.add_argument("--max-concurrent-requests", type=int, default=16, help="Maximum number of concurrent HTTP requests.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    run_embed_names_pipeline(
        input_path=args.input_path,
        output_path=args.output_path,
        service_url=args.service_url,
        max_concurrent_requests=args.max_concurrent_requests,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
