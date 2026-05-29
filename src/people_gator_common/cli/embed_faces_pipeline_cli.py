from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

import aiohttp
import dotenv
import tqdm

from people_gator_common.filesytem import glob, open_file, read_bytes

dotenv.load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI for face detection and embedding over a collection of images.")
    parser.add_argument(
        "--images-glob",
        type=str,
        default="images/*/**/*.*",
        help="Glob pattern to match input image files.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="embeddings/face_embeddings.jsonl",
        help="Path to write the JSONL output file with face embeddings.",
    )
    parser.add_argument(
        "--service-url",
        type=str,
        default="https://dev.people-gator-services.capturemyhand.com/api/face-image-pipeline/predict",
        help="URL of the face-image-pipeline prediction endpoint.",
    )
    parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=64,
        help="Maximum number of concurrent HTTP requests.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    return parser.parse_args()


async def detect_and_embed_on_image(
    path: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    service_url: str,
) -> dict:
    async with semaphore:
        image_bytes = await asyncio.to_thread(read_bytes, path)
        async with session.post(service_url, data={'input': image_bytes}) as response:
            result = await response.json()
        result['path'] = path
    return result


async def async_main(
    images_glob: str,
    output_path: str,
    service_url: str,
    max_concurrent_requests: int,
) -> None:
    images = glob(images_glob)
    logging.info(f"Found {len(images)} images to process.")
    with open_file(output_path, 'w') as f:
        def write_result(result: dict) -> None:
            f.write(json.dumps(result) + '\n')

        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(max_concurrent_requests)
            tasks = []
            for path in tqdm.tqdm(images):
                async with semaphore:
                    await asyncio.sleep(0.0)  # Handle task limit
                task = asyncio.create_task(
                    detect_and_embed_on_image(path, session, semaphore, service_url)
                )
                task.add_done_callback(lambda t: write_result(t.result()))
                tasks.append(task)
            logging.info(f"Created {len(tasks)} tasks for processing images.")
            await asyncio.gather(*tasks)
            logging.info("All tasks completed.")


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    asyncio.run(async_main(
        images_glob=args.images_glob,
        output_path=args.output_path,
        service_url=args.service_url,
        max_concurrent_requests=args.max_concurrent_requests,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
