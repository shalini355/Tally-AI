from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Tally AI API"
    database_url: str = "postgresql+psycopg://tally:tally@localhost:5432/tally"
    redis_url: str = "redis://localhost:6379/0"
    s3_bucket: str = "tally-ai-private"
    aws_region: str = "ap-south-1"
    max_upload_bytes: int = 50_000_000
    celery_task_max_retries: int = 5
    celery_task_backoff_max: int = 300
    idempotency_lock_seconds: int = 900


@lru_cache
def get_settings() -> Settings:
    return Settings()
