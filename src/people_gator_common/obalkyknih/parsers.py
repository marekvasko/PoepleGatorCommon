from collections.abc import Iterator
import logging
import multiprocessing
import os
from bs4 import BeautifulSoup
from bs4.filter import SoupStrainer
import pandas as pd
from people_gator_common.filesytem import ensure_parent_dir, exists, glob, open_file, read_text
from tqdm import tqdm
from multiprocessing import Pool

from people_gator_common.nlcr_aut.io import csv_escape

def extract_previewimage_from_html(
    path: str
) -> tuple[str, str | None]:
    """Return the `href` from a <link rel="previewimage"> tag, or None if not present."""
    html = read_text(path)
    soupstrainer = SoupStrainer('link', rel="previewimage")
    soup = BeautifulSoup(html, "lxml", parse_only=soupstrainer)
    for tag in soup.find_all('link', rel="previewimage"):
        href = tag.attrs.get("href")
        if not isinstance(href, str) or not href.strip():
            continue
        return path, href.strip()
    return path, None

def extract_previewimage_from_htmls(
    html_dir: str,
    output_csv: str,
    output_skipped_csv: str,
    *,
    num_workers: int = 8,
    show_progress: bool = True,
    diff_only: bool = True,
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
    
    ensure_parent_dir(output_csv)
    existing_entries = set()
    if diff_only:
        logger.info(
            "Diff-only mode enabled. Reading existing entries from output CSV: %s", output_csv,
            extra={
                "output_csv": str(output_csv),            }
        )
        if exists(output_csv):
            with open_file(output_csv, "r", encoding="utf-8") as f:
                existing_entries_df = pd.read_csv(f)
                existing_entries.update(
                    existing_entries_df["entity"].astype(str).tolist()
                )
        if exists(output_skipped_csv):
            with open_file(output_skipped_csv, "r", encoding="utf-8") as f:
                existing_skipped_entries_df = pd.read_csv(f)
                existing_entries.update(
                    existing_skipped_entries_df["entity"].astype(str).tolist()
                )    
        logger.info(
            "Loaded %d existing entries from output CSV for diffing", len(existing_entries),
            extra={
                "output_csv": str(output_csv),
                "existing_entries_count": len(existing_entries),
            }
        )
    

    logger.info(
        "Starting extraction of preview images from HTML files.",
        extra={
            "html_dir": str(html_dir),
            "output_csv": str(output_csv),
            "show_progress": show_progress,
            "diff_only": diff_only,
        }
    )
    
    if not diff_only or not exists(output_csv):
        with open_file(output_csv, "w", encoding="utf-8") as f:
            f.write("entity,value\n")
    if not diff_only or not exists(output_skipped_csv):
        with open_file(output_skipped_csv, "w", encoding="utf-8") as f:
            f.write("entity,value\n")

    with multiprocessing.get_context('spawn').Pool(num_workers) as pool:
        iterator = tqdm(
            [p for p in html_paths if p not in existing_entries], desc="Extracting preview images"
        ) if show_progress else iter(html_paths)
        imap = pool.imap(extract_previewimage_from_html, iterator)
        with open_file(output_csv, "a", encoding="utf-8") as f, open_file(output_skipped_csv, "a", encoding="utf-8") as skipped_f:
            for path, preview_image_url in imap:
                if preview_image_url is None:
                    skipped_f.write(f"{csv_escape(str(path))},\n")
                    data.append({
                        "entity": str(path),
                        "value": None,
                    })
                    continue
                f.write(f"{csv_escape(str(path))},{csv_escape(preview_image_url)}\n")
                data.append({
                    "entity": str(path),
                    "value": preview_image_url,
                })
    df = pd.DataFrame(data)
    return df
