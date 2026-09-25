from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DefaultSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        frozen=True,
        env_nested_delimiter="__",
    )


class Settings(DefaultSettings):
    """Application settings.

    Defaults target a local (host) run against `docker compose` services.
    compose.yml overrides the hosts for in-container communication.
    """

    app_version: str = "0.1.0"
    debug: bool = True
    environment: str = "development"
    service_name: str = "video-rag-api"

    # PostgreSQL configuration
    postgres_database_url: str = "postgresql+psycopg2://video_rag:video_rag_password@localhost:5432/video_rag"
    postgres_echo_sql: bool = False
    postgres_pool_size: int = 20
    postgres_max_overflow: int = 0

    # Redis configuration (db 0: cache; db 1/2: Celery broker/results)
    redis_url: str = "redis://localhost:6379/0"
    redis_socket_timeout: float = 2.0

    # Celery worker configuration
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    celery_task_time_limit: int = 60 * 60
    celery_task_soft_time_limit: int = 55 * 60

    # S3-compatible object storage (SeaweedFS locally, AWS S3 in production)
    s3_endpoint_url: str | None = "http://localhost:8333"
    s3_public_endpoint_url: str | None = None
    s3_access_key: str = "video_rag"
    s3_secret_key: str = "video_rag_secret"
    s3_region: str = "us-east-1"
    s3_bucket: str = "video-rag"
    s3_presigned_url_ttl: int = 3600

    # Uploads
    upload_max_mb: int = 500

    # Pexels (https://www.pexels.com/api/documentation/)
    pexels_api_key: str = ""
    pexels_base_url: str = "https://api.pexels.com"
    pexels_max_duration_sec: int = 60  # keep stock clips short: processing cost scales with duration
    pexels_max_height: int = 720  # download the best rendition at or below this height

    # Video processing (worker)
    scene_threshold: float = 0.3  # ffmpeg scene-change score (0-1); lower = more cuts
    visual_min_segment_sec: float = 1.0  # shots shorter than this merge into the previous one
    visual_max_segment_sec: float = 10.0  # longer shots are split so each segment has a representative keyframe
    frame_width: int = 640
    speech_window_sec: float = 15.0
    speech_stride_sec: float = 10.0  # window - stride = overlap, so a quote spanning a boundary is in one window

    # Speech-to-text (faster-whisper, runs in the worker)
    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_beam_size: int = 5

    # OpenSearch configuration
    opensearch_host: str = "http://localhost:9200"
    opensearch_index: str = "video_clips"

    # Ollama configuration
    ollama_host: str = "http://localhost:11434"
    ollama_models: str | list[str] = Field(default=["llama3.2:1b"])
    ollama_default_model: str = "llama3.2:1b"
    ollama_timeout: int = 300

    @field_validator("ollama_models", mode="before")
    @classmethod
    def parse_ollama_models(cls, v):
        if isinstance(v, str):
            return [model.strip() for model in v.split(",") if model.strip()]
        return v


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()
