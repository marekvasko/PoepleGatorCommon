from __future__ import annotations

import argparse

import dotenv

from people_gator_common.cli.helpers import configure_logging
from people_gator_common.pipelines.embed_faces import run_embed_faces_pipeline

dotenv.load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI for face detection and embedding over a collection of images.")
    parser.add_argument("--images-glob", type=str, default="images/*/**/*.*", help="Glob pattern to match input image files.")
    parser.add_argument("--output-path", type=str, default="embeddings/face_embeddings.jsonl", help="Path to write the JSONL output file with face embeddings.")
    parser.add_argument("--service-url", type=str, default="https://dev.people-gator-services.capturemyhand.com/api/face-image-pipeline/predict", help="URL of the face-image-pipeline prediction endpoint.")
    parser.add_argument("--max-concurrent-requests", type=int, default=64, help="Maximum number of concurrent HTTP requests.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    run_embed_faces_pipeline(
        images_glob=args.images_glob,
        output_path=args.output_path,
        service_url=args.service_url,
        max_concurrent_requests=args.max_concurrent_requests,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
