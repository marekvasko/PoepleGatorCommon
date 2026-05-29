from __future__ import annotations

import argparse
import asyncio
import logging
import sys

import aiohttp
import dotenv
import numpy as np
import pandas as pd
import tqdm

from people_gator_common.filesytem import open_file

dotenv.load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI for embedding person names via the sentence-transformer service.")
    parser.add_argument(
        "--input-path",
        type=str,
        default="embeddings/names.csv",
        help="Path to the CSV file containing a 'name' column.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="embeddings/names_embeddings.npy",
        help="Path to write the resulting embeddings as a NumPy .npy file.",
    )
    parser.add_argument(
        "--service-url",
        type=str,
        default="https://dev.people-gator-services.capturemyhand.com/api/sentence-transformer/predict",
        help="URL of the sentence-transformer prediction endpoint.",
    )
    parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=16,
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


async def async_main(
    input_path: str,
    output_path: str,
    service_url: str,
    max_concurrent_requests: int,
) -> None:
    with open_file(input_path, 'r') as f:
        names = pd.read_csv(f)

    async with aiohttp.ClientSession() as session:
        async def fetch_embedding(name: str, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore) -> dict:
            async with semaphore:
                payload = {'text': name}
                async with session.post(service_url, json=payload) as response:
                    return await response.json()

        semaphore = asyncio.Semaphore(max_concurrent_requests)
        tasks = []
        for name in tqdm.tqdm(names['name'].values, desc="Fetching embeddings"):
            async with semaphore:
                await asyncio.sleep(0.0001)  # Handle task limit
            tasks.append(asyncio.create_task(fetch_embedding(name, session, semaphore)))
        await asyncio.gather(*tasks)

    tasks_results = [task.result() for task in tasks]
    embedding_np = np.array([result['embedding'] for result in tasks_results])
    with open_file(output_path, 'wb') as f:
        np.save(f, embedding_np)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    asyncio.run(async_main(
        input_path=args.input_path,
        output_path=args.output_path,
        service_url=args.service_url,
        max_concurrent_requests=args.max_concurrent_requests,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
