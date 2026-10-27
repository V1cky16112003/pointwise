"""Request ids, the API key, the drain flag and metrics (src/serve_ops.py), tested on a tiny ASGI app with no model."""
import logging
import re
import threading
import traceback

import pytest

pytest.importorskip("starlette")
pytest.importorskip("httpx")

from starlette.applications import Starlette  # noqa: E402
from starlette.responses import JSONResponse, PlainTextResponse  # noqa: E402
from starlette.middleware import Middleware  # noqa: E402
from starlette.routing import Route  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from src import serve_ops as ops_mod  # noqa: E402
from src.serve_ops import Metrics, Ops, RequestContext, request_id_var  # noqa: E402


@pytest.fixture
def ops(monkeypatch):
    fresh = Ops()
    monkeypatch.setattr(ops_mod, "OPS", fresh)
    return fresh


def make_app():
    seen = {}

    async def echo(request):
        seen["rid"] = request_id_var.get()
        logging.getLogger("src.serve").info("inside the handler")
        return JSONResponse({"ok": True})

    async def boom(request):
        raise RuntimeError("secret detail")

    async def ready(request):
        return PlainTextResponse("up")

    # added as app middleware, as serve.py does: it sits inside Starlette's own error handler, so an unhandled error
    # reaches RequestContext first
    app = Starlette(routes=[Route("/v1/systemone", echo, methods=["POST"]), Route("/boom", boom), Route("/healthz", ready),
                            Route("/readyz", ready), Route("/metrics", ready)], middleware=[Middleware(RequestContext)])
    return app, seen


@pytest.fixture
def client(ops):
    app, seen = make_app()
    c = TestClient(app, raise_server_exceptions=False)
    c.seen = seen
    return c


def test_every_response_has_a_request_id_and_a_valid_caller_id_is_kept(client):
    generated = client.post("/v1/systemone").headers["x-request-id"]
    assert re.fullmatch(r"[0-9a-f]{32}", generated)
    assert client.post("/v1/systemone", headers={"X-Request-Id": "abc-123_x.9"}).headers["x-request-id"] == "abc-123_x.9"
    assert client.seen["rid"] == "abc-123_x.9"  # the handler saw the same id


@pytest.mark.parametrize("bad", ["", "has space", "semi;colon", "x" * 65, "new\nline", "<script>"])
def test_a_malformed_caller_id_is_replaced_not_echoed(client, bad):
    got = client.post("/v1/systemone", headers={"X-Request-Id": bad}).headers["x-request-id"]
    assert got != bad and re.fullmatch(r"[0-9a-f]{32}", got)


def test_log_lines_carry_the_request_id_and_no_body_or_query(client, caplog):
    ops_mod.install_request_id_logging()
    with caplog.at_level(logging.INFO, logger="src.serve"):
        client.post("/v1/systemone?secret=hunter2", json={"state": "my card number is 4111"}, headers={"X-Request-Id": "rid-1"})
    inside = [r for r in caplog.records if r.getMessage() == "inside the handler"]
    assert inside and inside[0].request_id == "rid-1"
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("request "))
    assert "status=200" in line and "route=/v1/systemone" in line
    assert "hunter2" not in caplog.text and "4111" not in caplog.text
    assert request_id_var.get() == "-"  # reset after the request


def test_probe_and_scrape_requests_are_not_logged_unless_they_fail(client, caplog):
    with caplog.at_level(logging.INFO, logger="src.serve"):
        for path in ("/healthz", "/readyz", "/metrics"):
            client.get(path)
    assert not any(r.getMessage().startswith("request ") for r in caplog.records)


def test_an_unhandled_error_is_counted_as_a_500_and_logged_without_its_text(client, ops, caplog):
    with caplog.at_level(logging.INFO, logger="src.serve"):
        got = client.get("/boom")
    assert got.status_code == 500
    text = ops.metrics.render()
    assert 'pointwise_requests_total{route="other",status="500"} 1' in text
    assert any("status=500" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)
    assert "secret detail" not in caplog.text


def test_without_a_key_everything_is_open(client):
    assert client.post("/v1/systemone").status_code == 200


def test_with_a_key_data_routes_need_it_and_probes_do_not(client, ops):
    ops.api_key = "s3cret-key"
    assert client.get("/healthz").status_code == 200 and client.get("/readyz").status_code == 200
    for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "s3cret-key"}, {"Authorization": "Basic s3cret-key"},
                    {"Authorization": "Bearer s3cret-key "}, {"Authorization": "Bearer"}):
        got = client.post("/v1/systemone", headers=headers)
        assert got.status_code == 401 and got.headers["www-authenticate"] == "Bearer", headers
        assert got.json() == {"detail": {"error": "unauthorized"}}
        assert "x-request-id" in got.headers  # a rejection still has its id
    assert client.post("/v1/systemone", headers={"Authorization": "Bearer s3cret-key"}).status_code == 200
    assert client.post("/v1/systemone", headers={"Authorization": "bearer s3cret-key"}).status_code == 200  # scheme: any case
    assert client.get("/metrics").status_code == 401  # metrics are not public when a key is set
    assert client.get("/metrics", headers={"Authorization": "Bearer s3cret-key"}).status_code == 200
    assert client.get("/docs").status_code == 401 and client.get("/nothing-here").status_code == 401  # deny by default


def test_a_wrong_key_does_not_reach_the_app_and_the_key_is_never_logged(client, ops, caplog):
    ops.api_key = "s3cret-key"
    with caplog.at_level(logging.DEBUG):
        client.post("/v1/systemone", headers={"Authorization": "Bearer s3cret-key-and-more"})
    assert "inside the handler" not in caplog.text and "s3cret" not in caplog.text


def test_draining_turns_away_data_routes_but_keeps_probes_and_metrics(client, ops):
    ops.draining = True
    got = client.post("/v1/systemone")
    assert got.status_code == 503 and got.headers["retry-after"] == "1" and got.json()["detail"]["error"] == "shutting_down"
    assert client.get("/healthz").status_code == 200 and client.get("/readyz").status_code == 200
    assert client.get("/metrics").status_code == 200


def test_metrics_counts_by_route_and_status_and_builds_a_cumulative_histogram():
    m = Metrics()
    for seconds in (0.01, 0.03, 0.03, 0.3, 100.0):
        m.observe("/v1/systemone", 200, seconds)
    m.observe("/v1/systemone", 503, 0.0)
    m.observe("other", 404, 0.001)
    text = m.render([("pointwise_queue_waiting", "gauge", "Waiting.", 3)])
    assert 'pointwise_requests_total{route="/v1/systemone",status="200"} 5' in text
    assert 'pointwise_requests_total{route="/v1/systemone",status="503"} 1' in text
    assert 'pointwise_requests_total{route="other",status="404"} 1' in text
    bucket = {le: int(n) for le, n in re.findall(r'systemone",le="([^"]+)"\} (\d+)', text)}
    assert bucket["0.025"] == 2 and bucket["0.05"] == 4 and bucket["0.4"] == 5  # 0.0 and 0.01 | + two 0.03s | + 0.3
    assert bucket["12.8"] == 5 and bucket["+Inf"] == 6  # the 100 s request only in +Inf
    assert 'pointwise_request_duration_seconds_count{route="/v1/systemone"} 6' in text
    assert re.search(r'pointwise_request_duration_seconds_sum\{route="/v1/systemone"\} 100\.37', text)
    assert "# TYPE pointwise_queue_waiting gauge\npointwise_queue_waiting 3\n" in text
    values = [int(n) for n in re.findall(r'systemone",le="[^+][^"]*"\} (\d+)', text)]
    assert values == sorted(values)  # buckets are cumulative


def test_metrics_stay_correct_when_many_threads_record_at_once():
    m = Metrics()

    def work():
        for _ in range(500):
            m.observe("/predict", 200, 0.01)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert 'pointwise_requests_total{route="/predict",status="200"} 4000' in m.render()


def test_only_known_routes_get_their_own_label(client, ops):
    client.get("/anything/1")
    client.get("/anything/2")
    text = ops.metrics.render()
    assert 'route="other",status="404"} 2' in text and "anything" not in text  # no label explosion from arbitrary paths


def test_two_authorization_headers_are_refused_even_if_one_is_right(client, ops):
    ops.api_key = "s3cret-key"
    for pair in ([("Authorization", "Bearer s3cret-key"), ("Authorization", "Bearer other")],
                 [("Authorization", "Bearer other"), ("Authorization", "Bearer s3cret-key")]):
        assert client.post("/v1/systemone", headers=pair).status_code == 401


def test_a_request_id_with_a_trailing_newline_is_replaced():
    assert ops_mod._ID.fullmatch("abc") and not ops_mod._ID.fullmatch("abc\n")  # `$` with match() would accept it


def test_an_unhandled_error_still_gets_a_request_id_a_generic_body_and_a_logged_id(client, ops, caplog):
    with caplog.at_level(logging.INFO, logger="src.serve"):
        got = client.get("/boom", headers={"X-Request-Id": "trace-me"})
    assert got.status_code == 500 and got.headers["x-request-id"] == "trace-me"
    assert got.json() == {"detail": {"error": "internal_error"}} and "secret detail" not in got.text
    failure = next(r for r in caplog.records if r.getMessage().startswith("unhandled RuntimeError"))
    assert failure.request_id == "trace-me" and failure.levelno == logging.ERROR  # the id is on the failure's own line
    assert "test_serve_ops.py" in failure.getMessage() and "boom" in failure.getMessage()  # where it happened...
    assert "secret detail" not in caplog.text  # ...not what it said
    assert 'pointwise_requests_total{route="other",status="500"} 1' in ops.metrics.render()


def test_a_cancelled_request_is_a_499_not_a_server_error(ops, caplog):
    import asyncio

    async def hang(scope, receive, send):
        raise asyncio.CancelledError

    messages = []

    async def send(message):
        messages.append(message)

    async def run():
        scope = {"type": "http", "path": "/v1/systemone", "method": "POST", "headers": []}
        await RequestContext(hang)(scope, None, send)

    with caplog.at_level(logging.INFO, logger="src.serve"), pytest.raises(asyncio.CancelledError):
        asyncio.run(run())
    assert 'pointwise_requests_total{route="/v1/systemone",status="499"} 1' in ops.metrics.render()
    assert not caplog.records or all(r.levelno < logging.WARNING for r in caplog.records)
    assert not any(r.levelno >= logging.WARNING for r in caplog.records) and ops.inflight == 0


def test_inflight_counts_data_requests_only_and_returns_to_zero(client, ops):
    for path in ("/healthz", "/readyz", "/metrics"):
        client.get(path)
    client.get("/boom")
    assert ops.inflight == 0


def test_failed_auth_is_logged_on_every_path_including_metrics_but_sampled(client, ops, caplog):
    ops.api_key = "s3cret-key"
    with caplog.at_level(logging.INFO, logger="src.serve"):
        client.get("/metrics")  # a quiet path, still logged: this is where key guessing would show up
        for _ in range(40):
            client.post("/v1/systemone", headers={"Authorization": "Bearer guess"})
    lines = [r for r in caplog.records if "status=401" in r.getMessage()]
    assert any("route=/metrics" in r.getMessage() for r in lines)
    assert client.post("/v1/systemone", headers={"Authorization": "Bearer guess"}).headers["connection"] == "close"
    assert len(lines) == 10 and all(r.levelno == logging.WARNING for r in lines)  # capped at 10 per window, not 41
    assert "guess" not in caplog.text


def test_the_suppressed_count_is_reported_when_the_window_rolls_over():
    log = ops_mod._RateLimitedLog(limit=2, window=10.0)
    shown = [log.allow(t)[0] for t in (0.0, 1.0, 2.0, 3.0, 4.0)]
    assert shown == [True, True, False, False, False]
    allowed, suppressed = log.allow(11.0)  # new window
    assert allowed and suppressed == 3


def test_websocket_connections_cannot_skip_the_key_or_the_drain(ops):
    import asyncio

    async def accept(scope, receive, send):
        await send({"type": "websocket.accept"})

    def connect(headers):
        sent = []

        async def receive():
            return {"type": "websocket.connect"}

        async def send(message):
            sent.append(message)

        asyncio.run(RequestContext(accept)({"type": "websocket", "path": "/ws", "headers": headers}, receive, send))
        return sent

    assert connect([])[0]["type"] == "websocket.accept"  # no key configured: open
    ops.api_key = "s3cret-key"
    assert connect([])[0] == {"type": "websocket.close", "code": 1008}
    assert connect([(b"authorization", b"Bearer wrong")])[0]["type"] == "websocket.close"
    assert connect([(b"authorization", b"Bearer s3cret-key")])[0]["type"] == "websocket.accept"
    ops.draining = True
    assert connect([(b"authorization", b"Bearer s3cret-key")])[0]["type"] == "websocket.close"


def test_a_cancel_after_the_response_started_keeps_the_real_status(ops):
    import asyncio

    async def answer_then_cancel(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        raise asyncio.CancelledError

    async def run():
        scope = {"type": "http", "path": "/v1/systemone", "method": "POST", "headers": []}
        await RequestContext(answer_then_cancel)(scope, None, lambda message: asyncio.sleep(0))

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())
    text = ops.metrics.render()
    assert 'status="200"} 1' in text and 'status="499"' not in text


def test_the_first_window_starts_at_the_first_event_whatever_the_clock_says():
    log = ops_mod._RateLimitedLog(limit=2, window=10.0)
    assert [log.allow(3.0)[0], log.allow(3.1)[0], log.allow(3.2)[0]] == [True, True, False]  # monotonic clock under 10 s
    assert log.allow(13.5) == (True, 1)  # next window reports the one that was left out


def test_a_refused_websocket_is_counted_and_logged(ops, caplog):
    import asyncio

    async def accept(scope, receive, send):
        await send({"type": "websocket.accept"})

    async def receive():
        return {"type": "websocket.connect"}

    async def send(message):
        pass

    ops.api_key = "s3cret-key"
    with caplog.at_level(logging.WARNING, logger="src.serve"):
        asyncio.run(RequestContext(accept)({"type": "websocket", "path": "/ws", "headers": []}, receive, send))
    assert 'route="other",status="403"} 1' in ops.metrics.render() and "websocket connection refused" in caplog.text


def test_a_websocket_that_disconnects_before_the_refusal_gets_no_close(ops):
    import asyncio

    async def accept(scope, receive, send):
        await send({"type": "websocket.accept"})

    sent = []

    async def receive():
        return {"type": "websocket.disconnect", "code": 1006}

    async def send(message):
        sent.append(message)

    ops.api_key = "s3cret-key"
    asyncio.run(RequestContext(accept)({"type": "websocket", "path": "/ws", "headers": []}, receive, send))
    assert sent == []  # nothing sent after the client has gone


def test_failure_logging_reads_no_source_files(client, monkeypatch):
    import linecache

    reads = []
    real = linecache.getlines
    monkeypatch.setattr(linecache, "getlines", lambda *a, **k: (reads.append(a), real(*a, **k))[1])
    client.get("/boom")
    assert reads == []  # walk_tb, not extract_tb: no disk I/O on the event loop while an error burst is being logged


def test_a_failure_after_the_response_started_is_logged_with_its_id_and_cannot_carry_caller_text_further(ops, caplog):
    import asyncio

    async def answer_then_fail(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        raise KeyError("caller text: card 4111")

    async def run():
        scope = {"type": "http", "path": "/v1/systemone", "method": "POST", "headers": [(b"x-request-id", b"late-1")]}
        await RequestContext(answer_then_fail)(scope, None, lambda message: asyncio.sleep(0))

    with caplog.at_level(logging.INFO, logger="src.serve"), pytest.raises(RuntimeError) as raised:
        asyncio.run(run())
    assert str(raised.value) == "KeyError after the response had started" and raised.value.__cause__ is None
    assert raised.value.__suppress_context__  # `from None`: the original, with its message, is not chained to it
    assert "4111" not in caplog.text and "4111" not in "".join(traceback.format_exception(raised.value))
    assert any(r.request_id == "late-1" and "unhandled KeyError" in r.getMessage() for r in caplog.records)
    assert 'status="200"} 1' in ops.metrics.render()  # the real status stands
    assert ops.inflight == 0


def test_refused_websockets_are_sampled_like_failed_auth(ops, caplog):
    import asyncio

    async def accept(scope, receive, send):
        await send({"type": "websocket.accept"})

    async def receive():
        return {"type": "websocket.connect"}

    async def send(message):
        pass

    ops.api_key = "s3cret-key"
    with caplog.at_level(logging.WARNING, logger="src.serve"):
        for _ in range(50):
            asyncio.run(RequestContext(accept)({"type": "websocket", "path": "/ws", "headers": []}, receive, send))
    assert caplog.text.count("websocket connection refused") == 10  # not 50
    assert 'route="other",status="403"} 50' in ops.metrics.render()  # all are still counted
