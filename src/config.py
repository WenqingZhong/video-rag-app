from typing import Literal

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
    environment: str = "development"  # "production": refuses to start with development secrets (unsafe_for_production)
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
    s3_access_key: str | None = "video_rag"  # empty on AWS: the instance's IAM role is used instead of keys
    s3_secret_key: str | None = "video_rag_secret"
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
    frame_max_side: int = 640  # keyframes fit within 640×640 (portrait and landscape alike)
    # Keyframes this uniform (grey-level std dev, 0–255) are blank: not captioned or embedded. Blank frames 0.0,
    # least varied real frame in the library 13.5 (a plain sky).
    frame_blank_max_stddev: float = 4.0
    speech_window_sec: float = 15.0
    speech_stride_sec: float = 10.0  # window - stride = overlap, so a quote spanning a boundary is in one window

    # Speech-to-text (faster-whisper, runs in the worker)
    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_beam_size: int = 5

    # Embedding service (CLIP, its own container)
    embedder_url: str = "http://localhost:8001"
    embedder_timeout: float = 60.0
    embedding_dim: int = 512  # must match the embedder's model (ViT-B-32); fixed in the index mapping
    embed_batch_size: int = 32

    # Captions (vision-language model served by Ollama)
    caption_model: str = "qwen2.5vl:3b"
    caption_max_side: int = 448  # downscale before captioning: ~4.5 s/frame on CPU instead of 10-30 s
    caption_timeout: float = 180.0
    caption_prompt: str = (
        "Describe this video frame in one factual sentence for a search index: "
        "the main subjects, what they are doing, and the setting."
    )

    # Request understanding (LLM → structured intent). Reuses the caption model: Ollama holds one model at a
    # time, so a separate text model would be reloaded every time captioning and /ask alternate.
    understanding_enabled: bool = True
    understanding_model: str = "qwen2.5vl:3b"
    understanding_timeout: float = 30.0

    # Question answering: the model answers only from these excerpts, citing them.
    qa_enabled: bool = True
    qa_max_excerpts: int = 8  # transcript chunks (~20 s) + keyframe captions shown to the model
    qa_timeout: float = 60.0

    # Chat agent: rules first, the model (router) only when the rules can't decide.
    agent_router_enabled: bool = True
    agent_router_timeout: float = 30.0
    chat_memory_ttl_sec: int = 24 * 3600
    chat_fetch_timeout_sec: int = 10 * 60  # give up waiting for downloaded Pexels videos to be processed

    # Who is asking (src/services/identity.py). Uploads are private to their owner and expire after a while.
    # SESSION_SECRET signs web session cookies and SERVICE_TOKEN lets the bot vouch for Telegram users:
    # both must be set to long random values outside local development.
    session_secret: str = "dev-only-session-secret-change-me"
    service_token: str = "dev-only-service-token-change-me"
    session_max_age_sec: int = 30 * 24 * 3600
    cookie_secure: bool = False  # True behind HTTPS
    upload_retention_days: int = 7
    # ADMIN_TOKEN (header x-admin-token) opens the admin, trace and ingestion endpoints, adds uploads to the shared
    # library, and skips the limits below (evaluation scripts, Airflow).
    admin_token: str = "dev-only-admin-token-change-me"

    # Limits per user (src/services/limits.py), counted in Redis per UTC day. A hosted model bills per token, so the
    # allowance is in tokens: every model call a user causes counts, including captioning their uploads.
    limits_enabled: bool = True
    limit_tokens_per_day: int = 50_000  # ~100 requests, or a few videos
    limit_ip_tokens_per_day: int = 200_000  # one address, however many anonymous sessions it opens
    limit_global_tokens_per_day: int = 2_000_000  # the whole app: a ceiling on the model bill
    limit_requests_per_min: int = 20
    limit_ip_requests_per_min: int = 60
    limit_uploads_per_day: int = 5
    limit_pexels_per_day: int = 3  # downloads of new stock footage
    upload_max_duration_sec: int = 600  # users' uploads; longer ones fail at the "probing" step

    # Telegram bot: a client of this API. Long-polls Telegram, so it needs no public URL.
    telegram_bot_token: str | None = None  # from @BotFather; only in .env (never committed)
    telegram_allowed_chat_ids: str = ""  # comma-separated; empty = anyone who finds the bot
    bot_api_url: str = "http://localhost:8000"  # where the bot reaches the API (compose: http://api:8000)

    # Token usage (every model call is written to the llm_calls table). The models run locally for free; cost is
    # an ESTIMATE: our token counts at a hosted model's list price. Change the reference here or in .env.
    usage_tracking_enabled: bool = True
    llm_price_reference: str = "Claude Haiku 4.5 list price"
    llm_price_input_per_mtok: float = 1.00  # USD per million input tokens (image tokens included)
    llm_price_output_per_mtok: float = 5.00  # USD per million output tokens

    # Tracing: every API request and Celery task is saved as a trace (timed spans) in the trace_spans table.
    tracing_enabled: bool = True
    trace_retention_days: int = 14  # older spans are deleted daily (Airflow DAG "maintenance")

    # Caching (Redis db 0). Keys include a fingerprint of the code and settings they depend on; answers also
    # include the index version, which every index write bumps. So TTLs are a safety net, not the invalidation.
    cache_enabled: bool = True
    understanding_cache_ttl_sec: int = 7 * 24 * 3600  # request → intent
    answer_cache_ttl_sec: int = 24 * 3600  # request (+ filters) → the full /ask answer

    # Clips (cut by the dedicated clip-worker, cached in S3 under clips/)
    clip_padding_sec: float = 0.75  # context around quoted words (and around the sentence of an answer or topic)
    # A clip is a moment, not a scene: visual clips are this long, centred on the matched keyframe (whole shots,
    # up to 10 s, felt like short videos rather than clips).
    clip_visual_sec: float = 5.0
    clip_max_sec: float = 8.0  # upper bound for any non-quote clip (a topic or answer sentence, or a fallback)
    clip_timeout: float = 60.0  # how long the API waits for the clip worker

    # OpenSearch configuration
    opensearch_host: str = "http://localhost:9200"
    opensearch_index: str = "video_segments"  # an ALIAS; it points at a versioned index (video_segments_v1)

    # Search
    search_phrase_slop: int = 2  # words the phrase may be stretched by (ASR inserting/dropping filler words)
    search_fuzzy_min_match: str = "75%"  # share of quote words that must (fuzzily) match in the last-resort strategy
    search_max_size: int = 50
    search_rrf_k: int = 60  # Reciprocal Rank Fusion constant (standard value; damps the weight of top ranks)
    search_vector_min_similarity: float = 0.15  # CLIP cosine below this is treated as "not a match"
    # Hybrid: a result found ONLY by vector search (no keyword support) must be at least this similar.
    # 0.20 chosen by a sweep (ADR 0002): false answers 5 → 2 of 12, MRR unchanged, one relevant video lost.
    search_vector_only_min_similarity: float = 0.20
    # Image queries (photo → keyframes). Photo-to-frame cosine runs far higher than text-to-frame: its own cut-off.
    # 0.57 chosen by make eval-images: 0.55–0.58 all give semantic hit@1 0.90 and 0/15 false answers (strongest
    # unrelated photo 0.550, weakest found match above it 0.584); 0.57 sits in the middle of that plateau.
    search_image_min_similarity: float = 0.57

    # Where the models run: "ollama" locally (the models above, by name) or "bedrock" on AWS.
    # Bedrock model ids come from the Bedrock console (Model catalog → the model's "Model ID", or an inference
    # profile id like "us.<model id>"); the account must have access to them. Credentials: the usual AWS chain.
    llm_provider: Literal["ollama", "bedrock"] = "ollama"
    bedrock_region: str = "us-east-1"
    bedrock_text_model_id: str | None = None  # understanding, answers, chat routing
    bedrock_vision_model_id: str | None = None  # keyframe captions (must accept images)

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


def unsafe_for_production(settings: Settings) -> list[str]:
    """What's wrong with these settings for a public deployment (empty = fine). Checked at startup when
    ENVIRONMENT=production, so a missing secret stops the deploy instead of shipping a forgeable cookie."""
    problems = []
    for name in ("session_secret", "service_token", "admin_token"):
        value = getattr(settings, name)
        if value.startswith("dev-only") or len(value) < 32:
            problems.append(f"{name.upper()} must be a random value of at least 32 characters")
    if not settings.cookie_secure:
        problems.append("COOKIE_SECURE must be true (the site is served over HTTPS)")
    if settings.debug:
        problems.append("DEBUG must be false")
    return problems


def check_production(settings: Settings) -> None:
    if settings.environment == "production" and (problems := unsafe_for_production(settings)):
        raise RuntimeError("Unsafe production settings: " + "; ".join(problems))


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()
