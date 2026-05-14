from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import tqdm
from requests import Session

from people_gator_common.obalkyknih.download import (
    download_obalkyknih_html,
    get_save_path,
)
from people_gator_common.rate_limiter import RateLimiter


@dataclass(frozen=True, slots=True)
class NLCRDownloadStats:
    total: int
    downloaded: int
    skipped_existing: int
    failed: int


def download_obalkyknih_html_batch(
    nlcr_aut_ids: Sequence[str],
    output_dir: Path,
    *,
    session: Session,
    rate_limiter: RateLimiter,
    show_progress: bool = True,
) -> NLCRDownloadStats:
    total = len(nlcr_aut_ids)
    downloaded = 0
    skipped_existing = 0
    failed = 0

    iterator: Sequence[str] | tqdm.tqdm[str]
    if show_progress:
        iterator = tqdm.tqdm(nlcr_aut_ids, total=total, desc="Downloading ObalkyKnih HTML")
    else:
        iterator = nlcr_aut_ids

    for nlcr_aut_id in iterator:
        save_path = get_save_path(nlcr_aut_id, save_root=output_dir)
        existed_before_download = save_path.exists()
        success = download_obalkyknih_html(
            nlcr_aut_id=nlcr_aut_id,
            output_dir=output_dir,
            session=session,
            rate_limiter=rate_limiter,
        )
        if success and existed_before_download:
            skipped_existing += 1
        elif success:
            downloaded += 1
        else:
            failed += 1

    return NLCRDownloadStats(
        total=total,
        downloaded=downloaded,
        skipped_existing=skipped_existing,
        failed=failed,
    )
