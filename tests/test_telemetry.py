"""src/telemetry.py: off by default, right labels and no text when on, flush on shutdown (no network: in-memory reader)."""
import pytest

pytest.importorskip("opentelemetry.sdk")
from opentelemetry.sdk.metrics.export import InMemoryMetricReader  # noqa: E402

from src import telemetry  # noqa: E402


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    telemetry.shutdown()
    yield
    telemetry.shutdown()


def points(reader):
    """{(metric name, sorted attributes): value or histogram count}"""
    out = {}
    data = reader.get_metrics_data()
    for rm in data.resource_metrics if data else []:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                for p in m.data.data_points:
                    out[(m.name, tuple(sorted(p.attributes.items())))] = p.value if hasattr(p, "value") else p.count
    return out


def test_it_is_off_without_an_endpoint_and_recording_does_nothing():
    telemetry.start(lambda: 0, lambda: 0)
    assert telemetry._provider is None
    telemetry.record_request("/v1/systemone", 200, 0.1)
    telemetry.record_overload()
    telemetry.record_cache(True)


def test_requests_overloads_cache_and_gauges_are_recorded_with_small_label_sets():
    reader = InMemoryMetricReader()
    telemetry.start(lambda: 3, lambda: 1.5, lambda: 42.0, reader=reader)
    telemetry.record_request("/v1/systemone", 200, 0.3)
    telemetry.record_request("/v1/systemone", 200, 0.1)
    telemetry.record_request("/v1/systemone", 422, 0.0)
    telemetry.record_overload()
    telemetry.record_cache(True)
    telemetry.record_cache(False)
    got = points(reader)
    assert got[("pointwise.requests", (("route", "/v1/systemone"), ("status", "200")))] == 2
    assert got[("pointwise.requests", (("route", "/v1/systemone"), ("status", "422")))] == 1
    assert got[("pointwise.request.duration", (("route", "/v1/systemone"),))] == 3
    assert got[("pointwise.overload_rejections", ())] == 1
    assert got[("pointwise.state_cache.lookups", (("result", "hit"),))] == 1
    assert got[("pointwise.state_cache.lookups", (("result", "miss"),))] == 1
    assert got[("pointwise.queue.waiting", ())] == 3 and got[("pointwise.gpu.busy", ())] == 1.5
    labels = {k for _, attrs in got for k, _ in attrs}
    assert got[("pointwise.gpu.active_time", ())] == 42.0
    assert got[("pointwise.container.uptime", ())] > 0
    assert labels <= {"route", "status", "result"}


def test_a_server_request_is_recorded_through_the_middleware():
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from src.serve_ops import RequestContext

    reader = InMemoryMetricReader()
    telemetry.start(lambda: 0, lambda: 0, reader=reader)
    app = Starlette(routes=[Route("/healthz", lambda r: PlainTextResponse("up"))], middleware=[Middleware(RequestContext)])
    TestClient(app).get("/healthz")
    TestClient(app).get("/secret-path-123")
    got = points(reader)
    assert got[("pointwise.requests", (("route", "/healthz"), ("status", "200")))] == 1
    assert got[("pointwise.requests", (("route", "other"), ("status", "404")))] == 1  # no label from arbitrary paths
    assert not any("secret-path" in str(k) for k in got)


def test_shutdown_flushes_once_and_stops_recording():
    reader = InMemoryMetricReader()
    telemetry.start(lambda: 0, lambda: 0, reader=reader)
    provider = telemetry._provider
    calls = []
    real = provider.shutdown
    provider.shutdown = lambda timeout_millis=0: calls.append(timeout_millis) or real(timeout_millis=timeout_millis)
    telemetry.shutdown()
    telemetry.shutdown()
    assert calls == [telemetry.FLUSH_TIMEOUT_MS]
    telemetry.record_request("/healthz", 200, 0.0)  # after shutdown: a no-op, not an error


def test_a_bad_setup_leaves_telemetry_off_instead_of_stopping_the_server(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://example.invalid")
    monkeypatch.setattr("opentelemetry.sdk.metrics.MeterProvider", lambda **kw: 1 / 0)
    telemetry.start(lambda: 0, lambda: 0)
    assert telemetry._provider is None


def test_recording_while_shutdown_clears_the_instruments_never_raises():
    telemetry.start(lambda: 0, lambda: 0, reader=InMemoryMetricReader())
    telemetry._duration = None  # what shutdown() on another thread can do between the check and the use
    telemetry.record_request("/healthz", 200, 0.0)


def test_the_real_exporter_path_builds_with_the_pinned_sdk(monkeypatch):
    """start() without a reader uses PeriodicExportingMetricReader + OTLPMetricExporter and the histogram bucket keyword;
    a version mismatch would otherwise only show as an empty dashboard after deploy."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:9")
    telemetry.start(lambda: 0, lambda: 0)
    assert telemetry._provider is not None
