"""Shared CLI helpers: rate-limiter args, logging setup."""

from __future__ import annotations

import argparse
import logging
import sys

from people_gator_common.rate_limiter import RateLimiter


def add_rate_limiter_args(
    parser: argparse.ArgumentParser,
    *,
    default_interval: float = 1.0,
    default_retries: int = 3,
    default_backoff: float = 5.0,
) -> None:
    """Add --rate-limit-interval/retries/backoff arguments to *parser*."""
    parser.add_argument(
        "--rate-limit-interval",
        type=float,
        default=default_interval,
        help=f"Minimum interval between requests in seconds (default: {default_interval}).",
    )
    parser.add_argument(
        "--rate-limit-retries",
        type=int,
        default=default_retries,
        help=f"Maximum number of retries (default: {default_retries}).",
    )
    parser.add_argument(
        "--rate-limit-backoff",
        type=float,
        default=default_backoff,
        help=f"Backoff time in seconds between retries (default: {default_backoff}).",
    )


def rate_limiter_from_args(args: argparse.Namespace) -> RateLimiter:
    """Construct a :class:`RateLimiter` from the standard rate-limit CLI args."""
    return RateLimiter(
        min_request_interval=float(args.rate_limit_interval),
        max_retries=int(args.rate_limit_retries),
        retry_backoff_seconds=float(args.rate_limit_backoff),
    )


def configure_logging(log_level: str) -> None:
    """Configure root logging with a consistent format."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


__all__ = [
    "add_rate_limiter_args",
    "rate_limiter_from_args",
    "configure_logging",
]
