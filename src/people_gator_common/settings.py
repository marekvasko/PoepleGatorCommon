from enum import Enum
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AliasChoices, BaseModel, Field, AliasPath

class LOG_LEVEL(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class S3Settings(BaseModel):
    """Settings for S3 storage."""
    enabled: bool = False
    bucket: str | None = None
    access_key_id: str | None = None
    secret_access_key: str | None = None
    endpoint_url: str | None = None

class LocalSettings(BaseModel):
    """Settings for local filesystem storage."""
    enabled: bool = False
    root: str | None = None

class LakeFSSettings(BaseModel):
    """Settings for LakeFS storage."""
    enabled: bool = False
    endpoint_url: str | None = None
    access_key_id: str | None = None
    secret_access_key: str | None = None
    repository: str | None = None
    branch: str | None = None    

class StorageSettings(BaseModel):
    """Supported storage types for ffspec filesystem configuration."""
    s3: S3Settings = S3Settings()
    local: LocalSettings = LocalSettings()
    lakefs: LakeFSSettings = LakeFSSettings()


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
        env_file=".env",
        extra='ignore',
    )
    log_level: LOG_LEVEL = LOG_LEVEL.INFO
    storage: StorageSettings = StorageSettings()
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



if __name__ == "__main__":
    settings = PeopleGatorSettings()
    print(settings.model_dump_json(indent=2))