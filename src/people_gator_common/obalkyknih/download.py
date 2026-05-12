from __future__ import annotations
import logging
from pathlib import Path
from requests import Session
from people_gator_common.errors import is_retryable_error
from people_gator_common.rate_limiter import RateLimiter, RateLimiterTooManyRequestsError

OBALKYKNIH_URL_TEMPLATE = "https://obalkyknih.cz/view_auth?auth_id={auth_id}"

def get_save_path(nlcr_aut_id: str, save_root: Path) -> Path:
    return save_root / f"{nlcr_aut_id}.html"

def download_obalkyknih_html(
    nlcr_aut_id: str,
    output_dir: Path,
    *,
    session: Session,
    rate_limiter: RateLimiter,
) -> bool:
    logger = logging.getLogger(__name__ + "." + download_obalkyknih_html.__name__)
    url = OBALKYKNIH_URL_TEMPLATE.format(auth_id=nlcr_aut_id)
    save_path = get_save_path(nlcr_aut_id, save_root=output_dir)
    if save_path.exists():
        logger.info(
            "HTML already exists, skipping download for nlcr_aut_id %s: %s", nlcr_aut_id, save_path,
            extra={
                "nlcr_aut_id": nlcr_aut_id, 
                "save_path": str(save_path)
            }
        )
        return True
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with rate_limiter as rl:
        while True:
            try:
                rl.request()
                response = session.get(url)
                response.raise_for_status()
                text = response.text
                logger.info(
                    "Successfully fetched obalky knih for nlcr_aut_id %s", 
                    nlcr_aut_id, 
                    extra={
                        "nlcr_aut_id": nlcr_aut_id,
                        "response_length": len(text)
                    }
                )
                with open(save_path, "w", encoding="utf-8") as file_handle:
                    file_handle.write(text)
                logger.info("Saved HTML for nlcr_aut_id %s to %s", nlcr_aut_id, save_path)
                return True
            except RateLimiterTooManyRequestsError as error:
                logger.warning(
                    "Too many requests error for nlcr_aut_id %s: %s", nlcr_aut_id, error,
                    extra={"nlcr_aut_id": nlcr_aut_id}
                )
                return False
            except Exception as e:
                logger.warning(
                    "Error fetching obalky knih for nlcr_aut_id %s: %s", nlcr_aut_id, e,
                    extra={"nlcr_aut_id": nlcr_aut_id}
                )
                if is_retryable_error(e):
                    logger.info(
                        "Retrying after error for nlcr_aut_id %s", nlcr_aut_id, 
                        extra={"nlcr_aut_id": nlcr_aut_id}
                    )
                    continue
                return False
