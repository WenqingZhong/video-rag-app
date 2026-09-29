"""A small tracer: nested, timed spans for one request or task, saved together when it ends.

    with start_trace("POST /api/v1/ask", service="api"):     # the root span; the trace id is the request id
        with span("understand") as s:                       # children nest by `with`
            s.set(tokens=448)

Code that runs outside a trace (tests, scripts) can still call span(): it does nothing.
The trace id travels to Celery tasks in a message header (see src/worker/tracing.py), so a clip cut or an
ingestion shows up in the trace of the request that caused it.
"""

import logging
import secrets
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


def new_id(nbytes: int = 8) -> str:
    return secrets.token_hex(nbytes)


@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_id: str | None
    name: str
    service: str
    started_at: datetime
    attributes: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None
    status: str = "ok"  # "ok" | "error" (an exception left the span)
    error: str | None = None
    _t0: float = field(default_factory=time.perf_counter, repr=False)

    def set(self, **attributes: Any) -> None:
        self.attributes.update({k: v for k, v in attributes.items() if v is not None})

    def fail(self, exc: BaseException) -> None:
        self.status, self.error = "error", f"{type(exc).__name__}: {exc}"[:2000]

    def end(self) -> None:
        if self.duration_ms is None:
            self.duration_ms = round((time.perf_counter() - self._t0) * 1000, 2)


@dataclass
class Trace:
    trace_id: str
    service: str
    spans: list[Span] = field(default_factory=list)
    discard: bool = False  # set by discard_trace(): nothing worth keeping happened (e.g. an empty poll)


class _NoSpan:
    """What span() yields outside a trace: accepts the same calls and records nothing."""

    span_id = None

    def set(self, **attributes: Any) -> None:
        pass

    def fail(self, exc: BaseException) -> None:
        pass


_trace: ContextVar[Trace | None] = ContextVar("trace", default=None)
_current: ContextVar[Span | None] = ContextVar("current_span", default=None)


def current_trace_id() -> str | None:
    trace = _trace.get()
    return trace.trace_id if trace else None


def discard_trace() -> None:
    """Don't save the current trace: a routine request (a poll that found nothing new) would only add noise."""
    trace = _trace.get()
    if trace is not None:
        trace.discard = True


def current_span_id() -> str | None:
    current = _current.get()
    return current.span_id if current else None


def _open(trace: Trace, name: str, parent_id: str | None, attributes: dict) -> Span:
    opened = Span(trace.trace_id, new_id(), parent_id, name, trace.service, datetime.now(UTC))
    opened.set(**attributes)
    trace.spans.append(opened)
    return opened


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Span | _NoSpan]:
    trace = _trace.get()
    if trace is None:
        yield _NoSpan()
        return
    parent = _current.get()
    opened = _open(trace, name, parent.span_id if parent else None, attributes)
    token = _current.set(opened)
    try:
        yield opened
    except BaseException as exc:
        opened.fail(exc)
        raise
    finally:
        opened.end()
        _current.reset(token)


class ActiveTrace:
    """A root span opened and closed by two separate calls (Celery's prerun/postrun signals, ASGI middleware)."""

    def __init__(self, trace: Trace, root: Span, tokens: tuple):
        self.trace, self.root, self._tokens = trace, root, tokens

    def finish(self, save: Callable[[list[Span]], None] | None = None) -> None:
        """Close the root span and leave the trace context; then save, unless the caller saves elsewhere."""
        self.root.end()
        trace_token, span_token = self._tokens
        try:
            _current.reset(span_token)
            _trace.reset(trace_token)
        except ValueError:  # finished in a different context than it started: nothing left to restore
            pass
        if save is not None:
            self.save(save)

    def save(self, save: Callable[[list[Span]], None]) -> None:
        if self.trace.discard:
            return
        try:
            save(self.trace.spans)
        except Exception as exc:  # noqa: BLE001 - losing a trace must never fail the request or task
            logger.warning("trace %s not saved: %s", self.trace.trace_id, exc)


def begin_trace(
    name: str, service: str, trace_id: str | None = None, parent_id: str | None = None, **attributes: Any
) -> ActiveTrace:
    trace = Trace(trace_id or new_id(16), service)
    root = _open(trace, name, parent_id, attributes)
    return ActiveTrace(trace, root, (_trace.set(trace), _current.set(root)))


@contextmanager
def start_trace(
    name: str,
    service: str,
    trace_id: str | None = None,
    save: Callable[[list[Span]], None] | None = None,
    **attributes: Any,
) -> Iterator[Span]:
    active = begin_trace(name, service, trace_id, **attributes)
    try:
        yield active.root
    except BaseException as exc:
        active.root.fail(exc)
        raise
    finally:
        active.finish(save)
