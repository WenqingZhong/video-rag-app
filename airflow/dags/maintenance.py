"""Daily housekeeping: delete trace spans older than the retention period (TRACE_RETENTION_DAYS, default 14).

Like the ingestion DAG, it only calls the API. Token usage (llm_calls) is never deleted: it is the cost history.
"""

import os
from datetime import UTC, datetime, timedelta

import requests
from airflow.sdk import dag, task

API_URL = os.environ.get("VIDEO_RAG_API_URL", "http://api:8000")


@dag(
    dag_id="maintenance",
    description="Delete old trace spans",
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
        response = requests.post(f"{API_URL}/api/v1/admin/cleanup-traces", timeout=300)
        response.raise_for_status()
        return response.json()

    cleanup_traces()


maintenance()
