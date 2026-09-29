"""Daily housekeeping:
- delete trace spans older than TRACE_RETENTION_DAYS (default 14);
- delete users' uploads older than UPLOAD_RETENTION_DAYS (default 7): index, files and rows. The library stays.

Like the ingestion DAG, it only calls the API. Token usage (llm_calls) is never deleted: it is the cost history.
"""

import os
from datetime import UTC, datetime, timedelta

import requests
from airflow.sdk import dag, task

API_URL = os.environ.get("VIDEO_RAG_API_URL", "http://api:8000")
ADMIN = {"x-admin-token": os.environ.get("ADMIN_TOKEN", "dev-only-admin-token-change-me")}


@dag(
    dag_id="maintenance",
    description="Delete old trace spans and expired uploads",
    schedule="@daily",
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
    tags=["maintenance", "observability"],
)
def maintenance():
    @task
    def cleanup_traces() -> dict:
        response = requests.post(f"{API_URL}/api/v1/admin/cleanup-traces", timeout=300, headers=ADMIN)
        response.raise_for_status()
        return response.json()

    @task
    def cleanup_uploads() -> dict:
        response = requests.post(f"{API_URL}/api/v1/admin/cleanup-uploads", timeout=600, headers=ADMIN)
        response.raise_for_status()
        return response.json()

    cleanup_traces()
    cleanup_uploads()


maintenance()
