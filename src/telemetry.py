"""OpenTelemetry metrics, pushed to Grafana Cloud (docs/superpowers/specs/2026-11-05-monitoring-design.md). No torch.

Off unless OTEL_EXPORTER_OTLP_ENDPOINT is set, so tests, CI, Docker and local runs record nothing and need no network.
The exporter reads its endpoint and credentials from the standard OTEL_EXPORTER_OTLP_* variables (a Modal Secret).
Labels are route, status and cache result only: small fixed sets, never request text, ids or user data.
Monitoring must never take serving down: a failed start leaves telemetry off, and export errors stay in the SDK's
background thread.
"""
from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger("src.serve")

BUCKETS = (0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8)  # seconds, as the old /metrics had
EXPORT_INTERVAL_MS = 15_000  # Modal's scaledown_window is 60 s, so a lost last batch costs at most a few seconds
FLUSH_TIMEOUT_MS = 3_000  # shutdown must stay inside POINTWISE_SHUTDOWN_GRACE_S

_BOOT = time.monotonic()  # imported before the model loads, so uptime includes the load time Modal also bills
_provider = None
_requests = _duration = _overloads = _cache = None


def start(waiting, busy_seconds, active_total=lambda: 0.0, reader=None) -> None:
    """Begin recording. The callables are read by the exporter every interval: requests waiting, age of the request holding
    the GPU, and total seconds the GPU has held requests since start. `reader` is for tests."""
    global _provider, _requests, _duration, _overloads, _cache
    if _provider is not None or (reader is None and not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")):
        return
    provider = None
    try:
        from opentelemetry.metrics import Observation
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.resources import Resource

        if reader is None:
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

            reader = PeriodicExportingMetricReader(OTLPMetricExporter(), export_interval_millis=EXPORT_INTERVAL_MS)
        provider = MeterProvider(metric_readers=[reader], resource=Resource.create({"service.name": "pointwise"}))
        meter = provider.get_meter("pointwise")
        requests = meter.create_counter("pointwise.requests", description="HTTP requests by route and status.")
        duration = meter.create_histogram("pointwise.request.duration", unit="s", description="Time to answer a request.",
                                          explicit_bucket_boundaries_advisory=BUCKETS)
        overloads = meter.create_counter("pointwise.overload_rejections", description="Requests turned away for overload.")
        cache = meter.create_counter("pointwise.state_cache.lookups", description="State cache lookups by result.")
        meter.create_observable_gauge("pointwise.queue.waiting", description="Requests waiting for the GPU.",
                                      callbacks=[lambda _: [Observation(waiting())]])
        meter.create_observable_counter("pointwise.container.uptime", unit="s", description="Seconds this server process has run (cost basis).",
                                        callbacks=[lambda _: [Observation(time.monotonic() - _BOOT)]])
        meter.create_observable_counter("pointwise.gpu.active_time", unit="s", description="Seconds the GPU held a request.",
                                        callbacks=[lambda _: [Observation(active_total())]])
        meter.create_observable_gauge("pointwise.gpu.busy", unit="s", description="How long the current request has held the GPU.",
                                      callbacks=[lambda _: [Observation(busy_seconds())]])
    except Exception as exc:  # a bad setting or missing package must not stop the server
        logger.warning(f"telemetry: not started ({type(exc).__name__}); serving without metrics")
        if provider is not None:  # stop the export thread a half-built provider may have started
            provider.shutdown(timeout_millis=FLUSH_TIMEOUT_MS)
        return
    _provider, _requests, _duration, _overloads, _cache = provider, requests, duration, overloads, cache
    logger.info("telemetry: pushing metrics")


# The record functions copy the global once: shutdown() runs on another thread and clears them.
def record_request(route: str, status: int, seconds: float) -> None:
    requests, duration = _requests, _duration
    if requests is not None and duration is not None:  # shutdown() clears them one by one
        requests.add(1, {"route": route, "status": str(status)})
        duration.record(seconds, {"route": route})


def record_overload() -> None:
    overloads = _overloads
    if overloads is not None:
        overloads.add(1)


def record_cache(hit: bool) -> None:
    cache = _cache
    if cache is not None:
        cache.add(1, {"result": "hit" if hit else "miss"})


def shutdown() -> None:
    """Send the last batch and stop. Blocks up to FLUSH_TIMEOUT_MS, so call it off the event loop."""
    global _provider, _requests, _duration, _overloads, _cache
    provider, _provider, _requests, _duration, _overloads, _cache = _provider, None, None, None, None, None
    if provider is not None:
        try:
            provider.shutdown(timeout_millis=FLUSH_TIMEOUT_MS)
        except Exception as exc:
            logger.warning(f"telemetry: final flush failed ({type(exc).__name__})")
