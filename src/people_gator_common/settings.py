from enum import Enum
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel

class LOG_LEVEL(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class StorageType(str, Enum):
    """Supported storage types for ffspec filesystem configuration."""
    LOCAL = "local"
    S3 = "s3"
    LAKEFS = "lakefs"

class RateLimiterSettings(BaseModel):
    """Settings for rate limiting."""
    enabled: bool = True
    max_retries: int
    retry_backoff_seconds: float
    min_request_interval: float

class PeopleGatorSettings(BaseSettings):
    """Settings for PeopleGator."""
    model_config = SettingsConfigDict(
        env_prefix="PEOPLEGATOR__",
        env_nested_delimiter="__",
    )
    log_level: LOG_LEVEL = LOG_LEVEL.INFO
    storage: StorageType = StorageType.LOCAL
    wikidata_rate_limiter: RateLimiterSettings | None = RateLimiterSettings(
        enabled=True,
        max_retries=3,
        retry_backoff_seconds=10.0,
        min_request_interval=5.0,
    )
    wikicommons_rate_limiter: RateLimiterSettings | None = RateLimiterSettings(
        enabled=True,
        max_retries=3,
        retry_backoff_seconds=5.0,
        min_request_interval=1.0,
    )
    obalkyknih_html_rate_limiter: RateLimiterSettings | None = RateLimiterSettings(
        max_retries=3,
        retry_backoff_seconds=5.0,
        min_request_interval=0.5,
    )
