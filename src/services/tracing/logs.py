"""Put the trace id on every log line, so the logs of one request can be found with grep."""

import logging

from src.services.tracing.tracer import current_trace_id

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(trace_id)s] %(message)s"


class TraceIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = current_trace_id() or "-"
        return True


def add_trace_ids(logger: logging.Logger, fmt: str = LOG_FORMAT) -> None:
    for handler in logger.handlers:
        handler.addFilter(TraceIdFilter())
        handler.setFormatter(logging.Formatter(fmt))
