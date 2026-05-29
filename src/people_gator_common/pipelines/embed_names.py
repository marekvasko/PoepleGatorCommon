"""Pipeline for embedding person names via an external sentence-transformer service."""

from __future__ import annotations

import asyncio
import logging

import aiohttp
import numpy as np
import tqdm

from people_gator_common.filesystem import open_file
from people_gator_common.utils import read_column_from_csv


async def _fetch_embedding(
    name: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    service_url: str,
) -> dict:
    async with semaphore:
        async with session.post(service_url, json={"text": name}) as response:
            return await response.json()


async def _async_embed_names(
    input_path: str,
    output_path: str,
    service_url: str,
    max_concurrent_requests: int,
) -> None:
    logger = logging.getLogger(__name__)
    names = read_column_from_csv(input_path, "name")
    logger.info("Embedding %d names from %s", len(names), input_path)
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(_fetch_embedding(name, session, semaphore, service_url))
            for name in tqdm.tqdm(names, desc="Fetching embeddings")
        ]
        await asyncio.gather(*tasks)
    embedding_np = np.array([task.result()["embedding"] for task in tasks])
    with open_file(output_path, "wb") as f:
        np.save(f, embedding_np)
    logger.info("Saved %d embeddings to %s", len(embedding_np), output_path)


def run_embed_names_pipeline(
    input_path: str,
    output_path: str,
    *,
    service_url: str,
    max_concurrent_requests: int,
) -> None:
    asyncio.run(_async_embed_names(input_path, output_path, service_url, max_concurrent_requests))
