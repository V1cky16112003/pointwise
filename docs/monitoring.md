# Monitoring Pointwise with Grafana Cloud

The server pushes OpenTelemetry metrics (OTLP over HTTP) to Grafana Cloud's free tier. Nothing scrapes the server, so a
scaled-to-zero container is never woken by monitoring. No alerts are set; you open the dashboard when you want to look.
Design: `docs/superpowers/specs/2026-11-05-monitoring-design.md`. Code: `src/telemetry.py`.

*Concepts.* A **counter** only goes up (requests). A **gauge** goes up and down (queue length). A **histogram** counts
values in buckets so you can read percentiles (p95 latency). **Labels** slice a metric (route, status); keep them to a
small fixed set, or storage explodes ("cardinality").

## What is sent

| Metric in Grafana (confirm exact names in Explore) | Type | Labels |
|---|---|---|
| `pointwise_requests_total` | counter | `route`, `status` |
| `pointwise_request_duration_seconds` (`_bucket/_sum/_count`) | histogram | `route` |
| `pointwise_overload_rejections_total` | counter | none |
| `pointwise_state_cache_lookups_total` | counter | `result` = hit or miss |
| `pointwise_queue_waiting` | gauge | none |
| `pointwise_container_uptime_seconds_total` | counter | none |
| `pointwise_gpu_active_time_seconds_total` | counter | none |
| `pointwise_gpu_busy_seconds` | gauge | none |

Request text never leaves the server; labels are fixed sets. Counters restart from zero when a container restarts, which
PromQL's `rate()` and `increase()` handle.

## Cost

Modal bills a container by the second while it is up, idle time and the 60 s scale-down window included. The server
reports its own uptime, so Grafana can estimate the GPU bill. Put the T4 price per second in a dashboard variable
`gpu_price_per_s` (read it from Modal's pricing page; it is not hard-coded anywhere, and prices change).

| Panel | PromQL |
|---|---|
| Estimated GPU cost, last 24 h (USD) | `increase(pointwise_container_uptime_seconds_total[24h]) * $gpu_price_per_s` |
| GPU utilization (share of billed time doing work) | `increase(pointwise_gpu_active_time_seconds_total[1h]) / increase(pointwise_container_uptime_seconds_total[1h])` |
| Cost per request (USD) | `increase(pointwise_container_uptime_seconds_total[24h]) * $gpu_price_per_s / increase(pointwise_requests_total{route="/v1/systemone"}[24h])` |

Utilization is the number to watch: a low value means most of the bill is idle time, which is the scale-down window and
cold starts. Add `sum()` around each metric if more than one container ever runs.

**This is an estimate, not the bill.** It misses the container's boot before Python starts, and it covers GPU time only
(not CPU, memory or the weights volume). Modal's billing page is the source of truth; compare the two once and note the
gap. Grafana Cloud's own usage page shows whether monitoring stays inside the free tier.

## One-time setup

1. In Grafana Cloud open your stack, then **Connections > OpenTelemetry (OTLP)** and note the **OTLP endpoint URL** and the
   **instance ID**. Create an **access token** there with the `metrics:write` scope.
2. Build the header value without echoing the token (run in your own terminal; paste the token when asked):
   ```bash
   read -rs "TOKEN?Grafana token: "; echo
   AUTH=$(printf '%s' "<instance-id>:$TOKEN" | base64)
   modal secret create pointwise-grafana \
     OTEL_EXPORTER_OTLP_ENDPOINT="<otlp-endpoint-url>" \
     OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic $AUTH"
   unset TOKEN AUTH
   ```
   The endpoint is the base URL only (the SDK adds `/v1/metrics`). zsh syntax shown; the token never goes in git.
3. `modal deploy deploy/modal_app.py`. The deploy fails if the Secret does not exist.

## Check it

1. Send a few requests (the demo page, or `curl` with the API key).
2. In Grafana: **Explore**, data source `grafanacloud-<stack>-prom`, query `pointwise_requests_total`. Metrics appear within
   about 30 s. If the names differ from the table above, fix the table.
3. Import the dashboard `docs/grafana-dashboard.json`: in Grafana go to **Dashboards > New > Import**, upload the file,
   pick your Prometheus data source (`grafanacloud-<stack>-prom`) for the `Prometheus` input, and click **Import**. It has
   rows for traffic, latency, queue and cache, and cost. The queries are the ones in the table above.
   Set the `T4 price per second` box at the top to Modal's current T4 rate (the file's default, 0.000164, is from memory,
   not checked). The time range controls the cost panels (cost in the range = uptime in the range x the price).
4. No data? Check the Modal logs for `telemetry: pushing metrics` (started) or `telemetry: not started` (a bad setting;
   serving continues without metrics), and the Secret's endpoint and token.

## Limits

- An idle service sends nothing, so "no data" cannot be told from "down". A "not ready" alert would need a scheduled
  synthetic request first; none is set.
- Export runs every 15 s and flushes at shutdown, bounded to 3 s. A hard kill can lose the last seconds.
- Free-tier limits (series, retention) are Grafana's; this uses well under 100 series.
