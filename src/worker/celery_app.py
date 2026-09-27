from celery import Celery

from src.config import get_settings

settings = get_settings()

celery_app = Celery(
    "video_rag",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["src.worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # Video processing jobs are long and heavy: acknowledge only after completion so a crashed
    # worker's job is redelivered, and never let one worker hoard queued jobs.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_track_started=True,
    task_time_limit=settings.celery_task_time_limit,
    task_soft_time_limit=settings.celery_task_soft_time_limit,
    result_expires=60 * 60 * 24,
    broker_connection_retry_on_startup=True,
    # Keep the health-check ping snappy when Redis is down.
    broker_transport_options={"socket_connect_timeout": 2, "socket_timeout": 2},
    # Interactive work (a user is waiting) gets its own queue and worker, so it never waits behind a long
    # video-processing job on the default "celery" queue.
    task_routes={"clip.cut": {"queue": "clips"}},
)
