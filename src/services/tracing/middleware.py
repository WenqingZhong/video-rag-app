"""Every API request becomes a trace. Its id is the request id, returned in the X-Request-ID header."""

import re
import time

import anyio

from src.services.metrics import observe_request
from src.services.tracing.tracer import begin_trace

REQUEST_ID_HEADER = "x-request-id"
# A caller may send its own id (to find its request later); anything odd is replaced by a fresh one.
_VALID_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
# Health checks run every few seconds and reading traces shouldn't create more: not traced.
UNTRACED = (
    "/api/v1/ping", "/api/v1/health", "/api/v1/traces", "/health", "/metrics", "/docs", "/redoc", "/openapi.json", "/app",
)  # fmt: skip


def _template(path: str, params: dict) -> str:
    """/api/v1/videos/abc with {"video_id": "abc"} → /api/v1/videos/{video_id}."""
    names = {str(v): k for k, v in params.items()}
    return "/".join(f"{{{names[part]}}}" if part in names else part for part in path.split("/"))


class TracingMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        if scope["type"] != "http" or path == "/" or path.startswith(UNTRACED):
            await self.app(scope, receive, send)
            return

        incoming = dict(scope.get("headers") or []).get(REQUEST_ID_HEADER.encode(), b"").decode("latin-1")
        trace_id = incoming if _VALID_ID.match(incoming) else None
        active = begin_trace(f"{scope['method']} {path}", service="api", trace_id=trace_id, path=path)
        root = active.root
        started = time.perf_counter()

        async def send_with_id(message):
            if message["type"] == "http.response.start":
                root.set(status_code=message["status"])
                if message["status"] >= 500:
                    root.status = "error"
                headers = [*message.get("headers", []), (REQUEST_ID_HEADER.encode(), active.trace.trace_id.encode())]
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        except BaseException as exc:
            root.fail(exc)
            raise
        finally:
            # Name by route template ("GET /api/v1/videos/{video_id}"), so traces of one endpoint group together.
            route = _template(path, scope.get("path_params") or {})
            root.name = f"{scope['method']} {route}"
            active.finish()
            # Metrics by route template too: one time series per endpoint, not one per video id.
            observe_request(scope["method"], route, root.attributes.get("status_code", 500), time.perf_counter() - started)
            store = getattr(scope["app"].state, "trace_store", None) if "app" in scope else None
            if store is not None:
                # After the response has been sent: saving never delays the caller, nor blocks the event loop.
                await anyio.to_thread.run_sync(active.save, store.save)
