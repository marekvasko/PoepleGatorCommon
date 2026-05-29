"""Pipeline for face detection and embedding over a collection of images."""

from __future__ import annotations

import asyncio
import json
import logging

import aiohttp
import tqdm

from people_gator_common.filesystem import glob, open_file, read_bytes


async def _detect_and_embed_on_image(
    path: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    service_url: str,
) -> dict:
    async with semaphore:
        image_bytes = await asyncio.to_thread(read_bytes, path)
        async with session.post(service_url, data={"input": image_bytes}) as response:
            result = await response.json()
        result["path"] = path
    return result


async def _async_embed_faces(
    images_glob: str,
    output_path: str,
    service_url: str,
    max_concurrent_requests: int,
) -> None:
    logger = logging.getLogger(__name__)
    images = glob(images_glob)
    logger.info("Found %d images to process.", len(images))
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    with open_file(output_path, "w") as f:
        async with aiohttp.ClientSession() as session:
            tasks = [
                asyncio.create_task(
                    _detect_and_embed_on_image(path, session, semaphore, service_url)
                )
                for path in tqdm.tqdm(images, desc="Scheduling face embed tasks")
            ]
            for task in tasks:
                task.add_done_callback(lambda t: f.write(json.dumps(t.result()) + "\n"))
            logger.info("Created %d tasks for processing images.", len(tasks))
            await asyncio.gather(*tasks)
    logger.info("All tasks completed.")


def run_embed_faces_pipeline(
    images_glob: str,
    output_path: str,
    *,
    service_url: str,
    max_concurrent_requests: int,
) -> None:
    asyncio.run(_async_embed_faces(images_glob, output_path, service_url, max_concurrent_requests))
