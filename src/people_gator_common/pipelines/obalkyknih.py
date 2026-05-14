from __future__ import annotations
from contextlib import nullcontext
import os
import pandas as pd
from people_gator_common.crawling import BatchDownloadStats
from people_gator_common.obalkyknih.fetchers import download_missing_obalkyknih_htmls
from people_gator_common.rate_limiter import RateLimiter
from people_gator_common.obalkyknih.parsers import extract_previewimage_from_htmls
from tqdm.contrib.logging import logging_redirect_tqdm

def run_obalkyknih_html_download_pipeline(
    input_csv: str,
    output_dir: str,
    *,
    rate_limiter: RateLimiter,
    auth_id_column: str = "entity",
    show_progress: bool = True,
) -> BatchDownloadStats:
    with logging_redirect_tqdm() if show_progress else nullcontext():
        return download_missing_obalkyknih_htmls(
            input_csv=input_csv,
            output_dir=output_dir,
            rate_limiter=rate_limiter,
            auth_id_column=auth_id_column,
            show_progress=show_progress,
        )

def run_obalkyknih_html_parse_pipeline(
    html_dir: str,
    output_csv: str,
    *,
    show_progress: bool = True,
) -> pd.DataFrame:
    with logging_redirect_tqdm() if show_progress else nullcontext():
        return extract_previewimage_from_htmls(
            html_dir=html_dir,
            output_csv=output_csv,
            show_progress=show_progress,
        )
