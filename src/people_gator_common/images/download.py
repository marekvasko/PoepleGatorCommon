from __future__ import annotations
import logging
from pathlib import Path
from urllib.parse import unquote, urlparse

from requests import Session
from people_gator_common.rate_limiter import RateLimiter, RateLimiterTooManyRequestsError
from people_gator_common.errors import is_retryable_error


def get_save_path(url: str, *, save_root: Path) -> Path | None:
    logger = logging.getLogger(__name__ + "." + get_save_path.__name__)

    parsed_url = urlparse(url)
    if not parsed_url.scheme.startswith("http"):
        logger.warning("Skipping non-http URL: %s", url)
        return None

    source = parsed_url.netloc
    image_path = Path(unquote(parsed_url.path).lstrip("/"))
    return save_root / source / image_path


def download_image(
    url: str,
    output_dir: Path,
    *,
    session: Session,
    rate_limiter: RateLimiter,
) -> bool:
    logger = logging.getLogger(__name__ + "." + download_image.__name__)

    save_path = get_save_path(url, save_root=output_dir)
    if save_path is None:
        return False

    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists():
        logger.info("Image already exists, skipping download: %s", save_path)
        return True

    # Use the provided RateLimiter instance for spacing and attempt tracking
    with rate_limiter as rl:
        while True:
            try:
                rl.request()
                response = session.get(url, timeout=30)
                response.raise_for_status()
                with open(save_path, "wb") as file_handle:
                    file_handle.write(response.content)
                logger.info("Downloaded image from %s to %s", url, save_path)
                return True
            except RateLimiterTooManyRequestsError as error:
                logger.warning("Too many requests error: %s", error)
                return False
            except Exception as error:
                logger.warning("Error: %s", error)
                if is_retryable_error(error):
                    logger.info("Retrying after error")
                    continue
                return False
