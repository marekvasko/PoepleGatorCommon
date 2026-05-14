from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
import logging
import random
import time
from collections.abc import Callable
from typing import TypeVar, Protocol

T = TypeVar("T")


class RateLimiterTooManyRequestsError(Exception):
    pass

class RateLimiterInstanceProtocol(Protocol):
    def request(self) -> None: ...
    async def arequest(self) -> None: ...

class RateLimiterProtocol(Protocol):
    def sign_for_delay(self) -> float: ...
    def __enter__(self) -> RateLimiterInstanceProtocol: ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...

@dataclass
class RateLimiterInstance(RateLimiterInstanceProtocol):
    total_attempts: int
    parent_rate_limiter: RateLimiter
    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger(__name__ + "." + RateLimiterInstance.__name__)
    )
    
    def _behaviour(self, wait_fn: Callable[[float], T]) -> T:
        self.total_attempts += 1
        if self.total_attempts >= self.parent_rate_limiter.max_retries:
            self.logger.warning("Reached maximum retry attempts (%d)", self.parent_rate_limiter.max_retries)
            raise RateLimiterTooManyRequestsError("Maximum retry attempts reached")
        
        backoff_seconds = self.parent_rate_limiter.retry_backoff_seconds * (self.total_attempts - 1)
        wait_seconds = self.parent_rate_limiter.sign_for_delay(backoff_seconds)
        
        if wait_seconds <= 0:
            self.logger.debug(
                "No need to wait to respect rate limit (attempt %d)",
                self.total_attempts,
                extra={
                    "wait_seconds": wait_seconds, 
                    "total_attempts": self.total_attempts,
                    "backoff_seconds": backoff_seconds,
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
                    "total_attempts": self.total_attempts,
                    "backoff_seconds": backoff_seconds,
                },
            )
            return wait_fn(wait_seconds)

    def request(self) -> None:
        self._behaviour(time.sleep)

    async def arequest(self) -> None:
        await self._behaviour(asyncio.sleep)

@dataclass
class RateLimiter(RateLimiterProtocol):
    max_retries: int
    retry_backoff_seconds: float
    min_request_interval: float
    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger(__name__ + "." + RateLimiter.__name__)
    )
    _next_execution_at: float = 0.0
    
    def sign_for_delay(self, backoff_seconds: float | None = None) -> float:
        now = time.monotonic()
        elapsed = now - self._next_execution_at
        wait_seconds = self.min_request_interval - elapsed
        expected_delay = max(wait_seconds, 0.0)
        self._next_execution_at = now + expected_delay + (backoff_seconds or 0.0)
        return expected_delay

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
