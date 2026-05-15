from enum import Enum

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel

class LOG_LEVEL(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class BaseStorageSettings(BaseModel):
    """Base settings for storage configurations."""
    enabled: bool = False

class S3Settings(BaseStorageSettings):
    """Settings for S3 storage."""
    bucket: str | None = None
    access_key_id: str | None = None
    secret_access_key: str | None = None
    endpoint_url: str | None = None

class LocalSettings(BaseStorageSettings):
    """Settings for local filesystem storage."""
    root: str | None = None

class LakeFSSettings(BaseStorageSettings):
    """Settings for LakeFS storage."""
    repository: str | None = None
    branch: str | None = None
    host: str | None = None
    api_key: str | None = None
    access_token: str | None = None

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



if __name__ == "__main__":
    settings = PeopleGatorSettings()
    print(settings.model_dump_json(indent=2))