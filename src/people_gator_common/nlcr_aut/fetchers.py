from __future__ import annotations

import pathlib
import requests
import tqdm


def download_and_extract_marc2021(url: str, output_path: pathlib.Path) -> pathlib.Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    size_from_header = response.headers.get("Content-Length")
    chunk_size = 8192
    total_size = (int(size_from_header) // (chunk_size // 8)) if size_from_header is not None else None
    with open(output_path, "wb") as f:
        for chunk in tqdm.tqdm(
            response.iter_content(chunk_size=chunk_size), desc="Downloading", total=total_size, unit="Blocks", unit_scale=True
        ):
            f.write(chunk)
    return output_path
