from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
import logging
import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")


class RateLimiterTooManyRequestsError(Exception):
    pass


@dataclass
class RateLimiterInstance:
    total_attempts: int
    parent_rate_limiter: RateLimiter
    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger(__name__ + "." + RateLimiterInstance.__name__)
    )
    
    def _compute_request_delay(self) -> tuple[float, float]:
        now = time.monotonic()
        elapsed = now - self.parent_rate_limiter.last_request_at
        wait_seconds = self.parent_rate_limiter.min_request_interval - elapsed
        wait_seconds += random.uniform(0, self.parent_rate_limiter.retry_backoff_seconds) * (
            self.total_attempts - 1
        )
        return now, wait_seconds
    
    def _behaviour(self, wait_fn: Callable[[float], T]) -> T:
        self.total_attempts += 1
        if self.total_attempts >= self.parent_rate_limiter.max_retries:
            self.logger.warning("Reached maximum retry attempts (%d)", self.parent_rate_limiter.max_retries)
            raise RateLimiterTooManyRequestsError("Maximum retry attempts reached")

        now, wait_seconds = self._compute_request_delay()
        self.parent_rate_limiter.last_request_at = now + max(wait_seconds, 0)
        
        if wait_seconds <= 0:
            self.logger.debug(
                "No need to wait to respect rate limit (attempt %d)",
                self.total_attempts,
                extra={
                    "wait_seconds": wait_seconds, 
                    "total_attempts": self.total_attempts
                },
            )
            return wait_fn(0.0)
        else:
            self.logger.debug(
                "Waiting %.2f seconds to respect rate limit (attempt %d)",
                wait_seconds,
                self.total_attempts,
                extra={
                    "wait_seconds": wait_seconds, 
                    "total_attempts": self.total_attempts
                },
            )
            return wait_fn(wait_seconds)

    def request(self) -> None:
        self._behaviour(time.sleep)

    async def arequest(self) -> None:
        await self._behaviour(asyncio.sleep)

@dataclass
class RateLimiter:
    max_retries: int
    retry_backoff_seconds: float
    min_request_interval: float
    last_request_at: float = 0.0
    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger(__name__ + "." + RateLimiter.__name__)
    )

    def __enter__(self) -> RateLimiterInstance:
        self.logger.debug(
            "Entering RateLimiter context", 
            extra={
                "max_retries": self.max_retries, 
                "retry_backoff_seconds": self.retry_backoff_seconds, 
                "min_request_interval": self.min_request_interval
            }
        )
        return RateLimiterInstance(
            total_attempts=0,
            parent_rate_limiter=self,
        )

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.logger.debug(
            "Exiting RateLimiter context",
            extra={
                "max_retries": self.max_retries, 
                "retry_backoff_seconds": self.retry_backoff_seconds, 
                "min_request_interval": self.min_request_interval
            }
        )
        # No special handling on exit, state is managed in RateLimiterInstance
