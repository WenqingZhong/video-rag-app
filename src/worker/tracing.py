"""Traces across the queue: the publisher puts its trace id in the message; the worker continues that trace.

Connected by importing this module (src/worker/celery_app.py does), in the API and in the workers.
"""

import logging

from celery.signals import after_setup_logger, before_task_publish, task_failure, task_postrun, task_prerun

from src.services.tracing import ActiveTrace, TraceStore, begin_trace, current_span_id, current_trace_id
from src.services.tracing.logs import add_trace_ids

logger = logging.getLogger(__name__)

UNTRACED_TASKS = {"system.ping"}  # health checks
_active: dict[str, ActiveTrace] = {}  # task id → its trace (a worker process runs one task at a time)


@before_task_publish.connect
def _add_trace_headers(headers: dict | None = None, **_) -> None:
    trace_id = current_trace_id()
    if headers is not None and trace_id:
        headers["trace_id"] = trace_id
        headers["parent_span_id"] = current_span_id()


def _service(task) -> str:
    hostname = getattr(task.request, "hostname", None) or ""
    return "clip-worker" if hostname.startswith("clips@") else "worker"


@task_prerun.connect
def _start(task_id: str, task, args=(), kwargs=None, **_) -> None:
    if task.name in UNTRACED_TASKS:
        return
    kwargs = kwargs or {}
    video_id = kwargs.get("video_id") or (args[0] if args and isinstance(args[0], str) else None)
    _active[task_id] = begin_trace(
        f"task {task.name}",
        service=_service(task),
        trace_id=getattr(task.request, "trace_id", None),  # None: started outside a request (e.g. by Airflow)
        parent_id=getattr(task.request, "parent_span_id", None),
        task_id=task_id,
        video_id=video_id,
        retry=task.request.retries or None,
    )


@task_failure.connect
def _failed(task_id: str, exception: BaseException, **_) -> None:
    if task_id in _active:
        _active[task_id].root.fail(exception)


@task_postrun.connect
def _finish(task_id: str, task, state: str | None = None, **_) -> None:
    active = _active.pop(task_id, None)
    if active is None:
        return
    active.root.set(state=state)
    from src.config import get_settings
    from src.worker.context import get_database  # the worker's own connection (imported late: API imports this too)

    active.finish(TraceStore(get_database()).save if get_settings().tracing_enabled else None)


@after_setup_logger.connect
def _log_trace_ids(logger: logging.Logger, **_) -> None:
    add_trace_ids(logger)
