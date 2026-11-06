# Pointwise monitoring: OpenTelemetry metrics to Grafana Cloud (free tier)

Status: DRAFT, waiting for the owner's approval. Build nothing until it is approved.

## Goal

See, in a Grafana Cloud dashboard, how the Modal-deployed Pointwise server behaves (traffic, errors, latency, queue,
cache) and while learning the stack companies use (OpenTelemetry, Prometheus data
model, PromQL, Grafana).

Replaces the Prometheus `/metrics` endpoint removed on 2026-11-05. No pull endpoint comes back: the server **pushes**,
so scale-to-zero on Modal is not defeated by a scraper waking the GPU.

## Design

**Module `src/telemetry.py`** (no torch, like `src/serve_ops.py`). It does nothing unless `OTEL_EXPORTER_OTLP_ENDPOINT`
is set, so tests, CI, Docker and local runs are unchanged.

When on, it builds an OTel `MeterProvider` with a `PeriodicExportingMetricReader` (OTLP over HTTP, every 15 s) and
exposes:

| OTel instrument | Prometheus name in Grafana | Type | Labels |
|---|---|---|---|
| `pointwise.requests` | `pointwise_requests_total` | counter | `route`, `status` |
| `pointwise.request.duration` (s) | `pointwise_request_duration_seconds_*` | histogram | `route` |
| `pointwise.overload_rejections` | `pointwise_overload_rejections_total` | counter | none |
| `pointwise.state_cache.lookups` | `pointwise_state_cache_lookups_total` | counter | `result` = hit or miss |
| `pointwise.queue.waiting` | `pointwise_queue_waiting` | observable gauge | none |
| `pointwise.container.uptime` (s) | `pointwise_container_uptime_seconds_total` | observable counter | none |
| `pointwise.gpu.active_time` (s) | `pointwise_gpu_active_time_seconds_total` | observable counter | none |
| `pointwise.gpu.busy` (s) | `pointwise_gpu_busy_seconds` | observable gauge | none |

(Exact Prometheus names depend on Grafana's OTLP translation; confirm in step 3 of the build and fix this table.)

**Hook points**
- `RequestContext` in `src/serve_ops.py`, where `observe(route, status, elapsed)` used to be: record requests + duration.
- `_overloaded` in `src/serve.py`: overload counter.
- `/v1/systemone` in `src/serve.py`, next to the existing `state_cache=hit|miss` log line: cache lookup counter.
- Gauges read `_waiting` and `_busy_since` through callbacks.

**Shutdown.** `provider.shutdown()` runs in the existing drain path, so the last batch is sent before Modal stops the
container. Modal's `scaledown_window` is 60 s, so a 15 s interval loses at most the final few seconds if the flush fails.

**Config and secrets.** Modal Secret `pointwise-grafana` holds `OTEL_EXPORTER_OTLP_ENDPOINT` and
`OTEL_EXPORTER_OTLP_HEADERS` (`Authorization=Basic <base64(instance_id:token)>`) plus `OTEL_SERVICE_NAME=pointwise`.
Added to `deploy/modal_app.py` beside `pointwise-key`. The token never enters git, logs or this chat. The owner
creates the secret.

**Dependencies.** `opentelemetry-sdk` and `opentelemetry-exporter-otlp-proto-http`, pinned in `requirements-serve.txt`
(versions checked at build time).

**Privacy.** Labels are `route`, `status` and `result` only: small fixed sets, never text, ids or user data. Same rule
as the logs.

## Cost

Estimated from the server's own uptime times a T4 price kept in a Grafana variable (never in code). Added 2026-11-05 at the
owner's request; panels in `docs/monitoring.md`. An estimate: GPU time only, not Modal's actual bill.

## Dashboard (built by hand in Grafana, documented in `docs/monitoring.md`)

Request rate, error rate by status, p50/p95 latency, queue waiting, cache hit rate, overload rejections.

**No alerts.** The owner does not want Grafana to notify them. The dashboard is looked at on demand. Alerting can be
added later, but note that with scale-to-zero an idle service sends no data, so a "not ready" alert would need a
scheduled synthetic request first.

**Scale to zero stays.** `min_containers=0` and `scaledown_window=60` in `deploy/modal_app.py` do not change. The
server only pushes while a container is up, and nothing outside polls it, so monitoring never wakes the GPU.

## Out of scope

Alerts, logs to Loki, traces, model-quality or drift tracking, synthetic checks, Sentry.

## Tests

- Without the env var, importing the server and handling requests records nothing and raises nothing.
- With an in-memory OTel reader: one 200, one 422 and one overload produce the right counters and labels; no text
  appears in any label.
- Shutdown calls the flush once.
- Break one hook on purpose and watch its test fail (project rule).

## Build order

1. Owner: put the Grafana OTLP endpoint and token into a Modal Secret `pointwise-grafana` (guided; token not shared in chat).
2. `src/telemetry.py` + hooks + tests.
3. Deploy, send traffic from the demo, check metric names in Grafana Explore, correct the table above.
4. Dashboard (no alerts); write `docs/monitoring.md`; update `docs/production.md`.

## Open items

- OTLP endpoint form (expected `https://otlp-gateway-<region>.grafana.net/otlp`): confirm from the Grafana stack page.
- Free-tier limits (series, retention): confirm on Grafana's site; the design uses well under 100 series.
