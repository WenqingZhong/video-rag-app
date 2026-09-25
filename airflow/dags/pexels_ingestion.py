"""Daily Pexels ingestion.

Airflow is the *scheduler* here, not the worker: each task makes one HTTP call to the Video RAG API,
which stores rows and queues Celery jobs. So this DAG needs no project dependencies, and every video
goes through exactly the same code path as a manual `POST /api/v1/videos/pexels`.
"""

import logging
import os
from datetime import UTC, datetime, timedelta

import requests
from airflow.sdk import Param, dag, get_current_context, task

API_URL = os.environ.get("VIDEO_RAG_API_URL", "http://api:8000")
REQUIRED_SERVICES = ("database", "redis", "object_storage", "worker")

logger = logging.getLogger(__name__)


@dag(
    dag_id="pexels_ingestion",
    description="Search Pexels for configured topics and queue new videos for processing",
    schedule="@daily",
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=2)},
    params={
        "topics": Param(
            ["dog", "cat", "city traffic", "ocean waves", "cooking", "people talking", "forest", "office meeting"],
            type="array",
            description="Search queries sent to Pexels",
        ),
        "per_topic": Param(3, type="integer", minimum=1, maximum=20, description="New videos to queue per topic"),
    },
    tags=["ingestion", "pexels"],
)
def pexels_ingestion():
    @task
    def check_api() -> None:
        """Fail fast (and retry) if the API or the services ingestion depends on are down."""
        response = requests.get(f"{API_URL}/api/v1/health", timeout=30)
        response.raise_for_status()
        services = response.json()["services"]
        down = [name for name in REQUIRED_SERVICES if services.get(name, {}).get("status") != "healthy"]
        if down:
            raise RuntimeError(f"Dependencies not healthy: {down}")

    @task
    def get_topics() -> list[str]:
        return list(get_current_context()["params"]["topics"])

    @task(max_active_tis_per_dag=2)  # stay gentle with the Pexels rate limit
    def ingest_topic(topic: str) -> dict:
        per_topic = get_current_context()["params"]["per_topic"]
        response = requests.post(f"{API_URL}/api/v1/videos/pexels", json={"query": topic, "count": per_topic}, timeout=120)
        response.raise_for_status()
        body = response.json()
        return {
            "topic": topic,
            "queued": len(body["queued"]),
            "skipped_existing": body["skipped_existing"],
            "skipped_unsuitable": body["skipped_unsuitable"],
            "video_ids": [v["id"] for v in body["queued"]],
        }

    @task
    def report(results: list[dict]) -> dict:
        summary = {
            "topics": len(results),
            "queued": sum(r["queued"] for r in results),
            "skipped_existing": sum(r["skipped_existing"] for r in results),
        }
        for r in results:
            logger.info("%-16s queued=%s skipped_existing=%s", r["topic"], r["queued"], r["skipped_existing"])
        logger.info("Summary: %s", summary)
        return summary

    topics = get_topics()
    check_api() >> topics
    report(ingest_topic.expand(topic=topics))


pexels_ingestion()
