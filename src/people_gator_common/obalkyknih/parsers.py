from collections.abc import Iterator
import logging
import os
from bs4 import BeautifulSoup
from bs4.filter import SoupStrainer
import pandas as pd
from people_gator_common.filesytem import ensure_parent_dir, glob, open_file, read_text
from tqdm import tqdm

from people_gator_common.nlcr_aut.io import csv_escape

def extract_previewimage_from_html(
    path: str
) -> str | None:
    """Return the `href` from a <link rel="previewimage"> tag, or None if not present."""
    html = read_text(path)
    soupstrainer = SoupStrainer('link', rel="previewimage")
    soup = BeautifulSoup(html, "lxml", parse_only=soupstrainer)
    for tag in soup.find_all('link', rel="previewimage"):
        href = tag.attrs.get("href")
        if not isinstance(href, str) or not href.strip():
            continue
        return href.strip()
    return None

def extract_previewimage_from_htmls(
    html_dir: str,
    output_csv: str,
    *,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Extract preview image URLs from a list of ObalkyKnih HTML files."""
    logger = logging.getLogger(__name__ + "." + extract_previewimage_from_htmls.__name__)
    data = []
    glob_pattern = os.path.join(html_dir, "**", "*.html")
    logger.info(
        "Extracting preview images from HTML files in directory: %s with pattern: %s", html_dir, glob_pattern,
        extra={
            "html_dir": str(html_dir),
            "glob_pattern": glob_pattern,
        }
    )
    html_paths = glob(os.path.join(html_dir, "**", "*.html"))
    logger.info(
        "Found %d HTML files to process", len(html_paths),
        extra={
            "html_dir": str(html_dir),
            "html_paths_count": len(html_paths),
            "glob_pattern": glob_pattern,
        }
    )
    iterator = tqdm(html_paths, desc="Extracting preview images") if show_progress else iter(html_paths)
    ensure_parent_dir(output_csv)
    with open_file(output_csv, "w", encoding="utf-8") as f:
        f.write("entity,value\n")
        for path in iterator:
            preview_image_url = extract_previewimage_from_html(path)
            if preview_image_url is None:
                continue
            f.write(f"{csv_escape(str(path))},{csv_escape(preview_image_url)}\n")
            data.append({
                "entity": str(path),
                "value": preview_image_url,
            })
    df = pd.DataFrame(data)
    return df
