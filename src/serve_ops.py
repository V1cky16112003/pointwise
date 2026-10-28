"""Operational plumbing for src/serve.py (Pointwise production step 3): request ids, the optional API key, the
shutdown drain flag, request metrics and the middleware that applies them. No torch, only ASGI, so it is tested without
a model.

Rules the middleware follows:
- Every request gets an id: the caller's X-Request-Id if it is 1-64 characters of [A-Za-z0-9._-], else a fresh one. It
  is echoed in the X-Request-Id response header and put on every log line made while handling the request.
- Logs carry metadata only (method, route, status, milliseconds), never the body or the query string.
- With an API key set, every path except /healthz and /readyz needs `Authorization: Bearer <key>`.
- While draining (shutdown has begun), data routes answer 503 + Retry-After; health, readiness and metrics still answer.
"""
from __future__ import annotations

import asyncio
import contextvars
import hmac
import logging
import math
import os
import re
import sys
import threading
import time
import traceback
import uuid
from bisect import bisect_left
from collections import Counter

from starlette.responses import JSONResponse

logger = logging.getLogger("src.serve")

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")  # used with fullmatch: `$` would also accept a trailing newline

OPEN_PATHS = frozenset({"/healthz", "/readyz"})  # probes never carry the key
DRAIN_OPEN_PATHS = OPEN_PATHS | {"/metrics"}
QUIET_PATHS = frozenset({"/healthz", "/readyz", "/metrics"})  # not logged per request unless they fail
ROUTES = frozenset({"/v1/systemone", "/predict", "/healthz", "/readyz", "/metrics"})  # anything else is "other"
# POINTWISE_SHUTDOWN_GRACE_S is split so the whole shutdown stays within it. Uvicorn sleeps a fixed 0.1 s outside its own
# timeout and the process needs a moment to close, so SHUTDOWN_MARGIN_S is held back; the rest is split: uvicorn waits for
# open connections for UVICORN_SHARE of it (a whole number of seconds, as uvicorn takes), the server's own drain gets
# exactly what is left. Keep the grace below the orchestrator's kill timeout (Kubernetes: 30 s by default).
DEFAULT_SHUTDOWN_GRACE_S = 25.0
UVICORN_SHARE = 0.8
SHUTDOWN_MARGIN_S = 1.0
MIN_GRACE_S = 3.0  # below this uvicorn's whole second plus the margin would overshoot the budget


def split_grace(grace: float) -> tuple[int, float]:
    """(seconds uvicorn waits for open connections, seconds the server's drain waits afterwards).
    The two plus SHUTDOWN_MARGIN_S add up to grace."""
    budget = grace - SHUTDOWN_MARGIN_S
    uvicorn_s = max(1, math.floor(budget * UVICORN_SHARE))
    return uvicorn_s, max(0.0, budget - uvicorn_s)


def env_float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        value = float(os.environ.get(name, default))
    except ValueError:
        sys.exit(f"FATAL: {name} must be a number, got {os.environ[name]!r}.")
    if not (0 < value <= threading.TIMEOUT_MAX and value >= minimum):  # also rejects inf and nan
        low = f"{minimum:g}" if minimum else "0 (exclusive)"
        sys.exit(f"FATAL: {name} must be a number from {low} up to {threading.TIMEOUT_MAX:g}, got {value}.")
    return value


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, "1" if default else "0")
    if value not in ("0", "1"):
        sys.exit(f"FATAL: {name} must be 0 or 1, got {value!r}.")
    return value == "1"


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        sys.exit(f"FATAL: {name} must be an integer, got {os.environ[name]!r}.")


class Metrics:
    """Request counters and a latency histogram in the Prometheus text format, with no dependency."""

    BUCKETS = (0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8)  # seconds

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: Counter = Counter()  # (route, status) -> count
        self._latency: dict[str, list[float]] = {}  # route -> per-bucket counts (last = overflow), then the sum

    def observe(self, route: str, status: int, seconds: float) -> None:
        with self._lock:
            self._requests[(route, status)] += 1
            row = self._latency.setdefault(route, [0] * (len(self.BUCKETS) + 1) + [0.0])
            row[bisect_left(self.BUCKETS, seconds)] += 1
            row[-1] += seconds

    def render(self, extra=()) -> str:
        """The text exposition. `extra` is (name, "gauge" | "counter", help, value) rows owned by the server."""
        with self._lock:
            requests = sorted(self._requests.items())
            latency = {route: list(row) for route, row in sorted(self._latency.items())}
        out = ["# HELP pointwise_requests_total HTTP requests by route and status.",
               "# TYPE pointwise_requests_total counter"]
        out += [f'pointwise_requests_total{{route="{route}",status="{status}"}} {n}' for (route, status), n in requests]
        out += ["# HELP pointwise_request_duration_seconds Time to answer a request, by route.",
                "# TYPE pointwise_request_duration_seconds histogram"]
        for route, row in latency.items():
            cumulative = 0
            for bound, n in zip(self.BUCKETS, row):
                cumulative += n
                out.append(f'pointwise_request_duration_seconds_bucket{{route="{route}",le="{bound:g}"}} {cumulative}')
            total = cumulative + row[len(self.BUCKETS)]
            out += [f'pointwise_request_duration_seconds_bucket{{route="{route}",le="+Inf"}} {total}',
                    f'pointwise_request_duration_seconds_sum{{route="{route}"}} {row[-1]:.6f}',
                    f'pointwise_request_duration_seconds_count{{route="{route}"}} {total}']
        for name, kind, help_text, value in extra:
            out += [f"# HELP {name} {help_text}", f"# TYPE {name} {kind}", f"{name} {value}"]
        return "\n".join(out) + "\n"


class Ops:
    """What the middleware reads at request time. The server sets api_key at startup and draining at shutdown."""

    def __init__(self) -> None:
        self.api_key: str | None = None
        self.draining = False
        self.signalled = False  # a shutdown signal has arrived; unlike draining it survives the app's startup
        self.inflight = 0  # data requests being handled right now (touched only on the event loop)
        self.drain_wait: float | None = None  # set by serve_main to its share; None (plain uvicorn) means the whole grace
        self.metrics = Metrics()
        self.auth_log = _RateLimitedLog()

    def authorised(self, raw_headers) -> bool:
        """True for exactly one `Authorization: Bearer <key>` header (scheme case-insensitive, key compared in constant
        time). Two Authorization headers are refused: a proxy and this server could otherwise judge different ones."""
        values = [v for k, v in raw_headers if k == b"authorization"]
        if self.api_key is None or len(values) != 1:
            return False
        scheme, _, token = values[0].partition(b" ")
        return scheme.lower() == b"bearer" and hmac.compare_digest(token, self.api_key.encode())

    def reject(self, path: str, raw_headers):
        """A ready-made response if this request must not reach the app (draining, or a missing or wrong key), else None."""
        if self.draining and path not in DRAIN_OPEN_PATHS:
            return JSONResponse({"detail": {"error": "shutting_down"}}, status_code=503,
                                headers={"Retry-After": "1", "Connection": "close"})
        if self.api_key is not None and path not in OPEN_PATHS and not self.authorised(raw_headers):
            # Connection: close, because the body of a refused request is not read and keep-alive would hold the socket
            return JSONResponse({"detail": {"error": "unauthorized"}}, status_code=401,
                                headers={"WWW-Authenticate": "Bearer", "Connection": "close"})
        return None


class _RateLimitedLog:
    """At most `limit` lines per `window` seconds, then one summary of how many were left out. For events an outsider
    can trigger at will, such as failed logins."""

    def __init__(self, limit: int = 10, window: float = 10.0) -> None:
        self.limit, self.window = limit, window
        self._start: float | None = None
        self._count = 0

    def allow(self, now: float) -> tuple[bool, int]:
        """(log this one?, how many were suppressed in the window that just ended)."""
        if self._start is None or now - self._start >= self.window:
            suppressed = max(0, self._count - self.limit)
            self._start, self._count = now, 0
        else:
            suppressed = 0
        self._count += 1
        return self._count <= self.limit, suppressed


OPS = Ops()


class RequestContext:
    """Outermost app middleware: request id, key and drain checks, per-request log line, metrics."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket":  # no websocket routes exist; if one is ever added it must not bypass the checks
            if OPS.reject(scope["path"], scope["headers"]) is not None:
                OPS.metrics.observe("other", 403, 0.0)  # a close before accept reaches the client as an HTTP 403
                show, suppressed = OPS.auth_log.allow(time.monotonic())  # as for HTTP 401s: anyone can cause these
                if suppressed:
                    logger.warning(f"{suppressed} further refused requests in the last window were not logged")
                if show:
                    logger.warning("websocket connection refused (missing or wrong key, or draining)")
                if (await receive())["type"] == "websocket.connect":  # not if the client has already gone
                    await send({"type": "websocket.close", "code": 1008})
                return
            return await self.app(scope, receive, send)
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        ops, path, method = OPS, scope["path"], scope["method"]
        route = path if path in ROUTES else "other"
        rid = dict(scope["headers"]).get(b"x-request-id", b"").decode("latin-1")
        if not _ID.fullmatch(rid):
            rid = uuid.uuid4().hex
        token = request_id_var.set(rid)
        counted = path not in QUIET_PATHS
        ops.inflight += counted
        start, status, started = time.monotonic(), 500, False  # 500 stands unless a response starts

        async def tracking_send(message):
            nonlocal status, started
            if message["type"] == "http.response.start":
                status, started = message["status"], True
                message = {**message, "headers": [*message.get("headers", []), (b"x-request-id", rid.encode())]}
            await send(message)

        try:
            early = ops.reject(path, scope["headers"])
            if early is not None:
                await early(scope, receive, tracking_send)
            else:
                await self.app(scope, receive, tracking_send)
        except asyncio.CancelledError:  # shutdown cancelled the request, or the client went away: not a server error
            if not started:  # once a response has started, its real status stands
                status = 499
            raise
        except Exception as exc:
            # Where it happened, never what it said: the message could hold caller text. The id is still set.
            frames = " <- ".join(f"{os.path.basename(frame.f_code.co_filename)}:{lineno} {frame.f_code.co_name}"
                                 for frame, lineno in reversed(list(traceback.walk_tb(exc.__traceback__))[-6:]))  # no source read
            logger.error(f"unhandled {type(exc).__name__} on route={route} at {frames}")
            if started:  # too late for a 500; fail the connection, with a message that cannot hold caller text
                raise RuntimeError(f"{type(exc).__name__} after the response had started") from None
            status = 500
            await JSONResponse({"detail": {"error": "internal_error"}}, status_code=500)(scope, receive, tracking_send)
        finally:
            elapsed = time.monotonic() - start
            ops.inflight -= counted
            ops.metrics.observe(route, status, elapsed)
            line = f"request method={method} route={route} status={status} ms={1000 * elapsed:.0f}"
            if status == 401:  # anyone can cause these, on any path: sample them, and never skip the quiet paths
                show, suppressed = ops.auth_log.allow(time.monotonic())
                if suppressed:
                    logger.warning(f"{suppressed} further failed-auth requests in the last window were not logged")
                if show:
                    logger.warning(line)
            elif counted or status >= 500:
                logger.log(logging.WARNING if status >= 500 else logging.INFO, line)
            request_id_var.reset(token)


class _RequestIdFilter(logging.Filter):
    """Gives a record the current request's id as `request_id` ("-" outside a request), unless it already has one."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):  # a caller's extra={"request_id": ...} wins
            record.request_id = request_id_var.get()
        return True


def install_request_id_logging() -> None:
    """Put the request id on every record logged through the `src.serve` logger, so a log format can use it. A filter on
    that logger, not a global record factory, so `extra={"request_id": ...}` still works anywhere in the process."""
    target = logging.getLogger("src.serve")
    if not any(isinstance(f, _RequestIdFilter) for f in target.filters):
        target.addFilter(_RequestIdFilter())
