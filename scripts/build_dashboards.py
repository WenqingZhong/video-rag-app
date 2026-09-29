"""Generate the Grafana dashboards (infra/grafana/dashboards/*.json). Grafana loads them at startup.

    make dashboards      # after editing a panel here; then restart grafana (or wait: it re-reads every 10 s)

Two dashboards:
  Tokens & cost          Postgres llm_calls: what the model calls consumed, the estimated cost, what caching saved
  Requests & latency     Prometheus (live request rates, latency, /ask results, cache hits)
                         + Postgres trace_spans (time per step, slowest traces, errors)
"""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "infra" / "grafana" / "dashboards"
PG = {"type": "grafana-postgresql-datasource", "uid": "video-rag-postgres"}
PROM = {"type": "prometheus", "uid": "video-rag-prometheus"}
API_URL = "http://localhost:8000"

# llm_calls rows that consumed tokens (cache hits are rows too, with 0 tokens), filtered by the dashboard's origin
CALLS = "FROM llm_calls WHERE $__timeFilter(created_at) AND origin IN ($origin)"
SPENT = f"{CALLS} AND outcome <> 'cache_hit'"
HITS = f"{CALLS} AND outcome = 'cache_hit'"


def sql(query: str, fmt: str = "table") -> dict:
    return {
        "refId": "A",
        "datasource": PG,
        "rawSql": " ".join(query.split()),
        "format": fmt,
        "rawQuery": True,
        "editorMode": "code",
    }


def prom(expr: str, legend: str = "", ref: str = "A") -> dict:
    return {"refId": ref, "datasource": PROM, "expr": expr, "legendFormat": legend, "range": True}


def panel(kind: str, title: str, pos: tuple[int, int, int, int], targets: list[dict], description: str = "", **extra) -> dict:
    x, y, w, h = pos
    datasource = targets[0]["datasource"]
    return {
        "type": kind,
        "title": title,
        "description": description,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "datasource": datasource,
        "targets": targets,
        "fieldConfig": {"defaults": extra.pop("defaults", {}), "overrides": extra.pop("overrides", [])},
        "options": extra.pop("options", {}),
        **extra,
    }


def stat(title, pos, target, unit="short", decimals=None, description="") -> dict:
    if "expr" in target:  # one number for the whole time range: an instant query at "now" over [$__range]
        target = {**target, "range": False, "instant": True}
    defaults = {"unit": unit, "color": {"mode": "fixed", "fixedColor": "blue"}}
    if decimals is not None:
        defaults["decimals"] = decimals
    options = {
        "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
        "colorMode": "value",
        "graphMode": "none",
        "textMode": "value",
    }
    return panel("stat", title, pos, [target], description, defaults=defaults, options=options)


def timeseries(title, pos, targets, unit="short", description="", stacked=False, bars=False) -> dict:
    custom = {"drawStyle": "bars" if bars else "line", "fillOpacity": 60 if bars else 10, "lineWidth": 1, "showPoints": "never"}
    if stacked:
        custom["stacking"] = {"mode": "normal", "group": "A"}
    options = {"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi", "sort": "desc"}}
    return panel("timeseries", title, pos, targets, description, defaults={"unit": unit, "custom": custom}, options=options)


def table(title, pos, target, description="", overrides=None) -> dict:
    return panel("table", title, pos, [target], description, overrides=overrides or [], options={"showHeader": True})


def column(name: str, **properties) -> dict:
    return {"matcher": {"id": "byName", "options": name}, "properties": [{"id": k, "value": v} for k, v in properties.items()]}


def dashboard(uid: str, title: str, description: str, panels: list[dict], variables: list[dict], time_from: str) -> dict:
    for i, p in enumerate(panels, start=1):
        p["id"] = i
    return {
        "uid": uid,
        "title": title,
        "description": description,
        "tags": ["video-rag"],
        "timezone": "browser",
        "schemaVersion": 41,
        "time": {"from": time_from, "to": "now"},
        "refresh": "30s",
        "templating": {"list": variables},
        "panels": panels,
        "links": [
            {"title": "Tokens & cost", "type": "link", "url": "/d/video-rag-tokens"},
            {"title": "Requests & latency", "type": "link", "url": "/d/video-rag-requests"},
        ],
    }


ORIGIN = {
    "name": "origin",
    "label": "Origin",
    "type": "custom",
    "query": "api,worker,eval",
    "multi": True,
    "includeAll": True,
    "current": {"text": ["api", "worker"], "value": ["api", "worker"]},
    "description": "api: /ask requests · worker: captions during ingestion · eval: evaluation runs",
}


def tokens_and_cost() -> dict:
    per_call_columns = [
        column("cost per 1000 calls", unit="currencyUSD", decimals=3),
        column("cost", unit="currencyUSD", decimals=4),
        column("avg reading prompt", unit="s", decimals=2),
    ]
    video_columns = [column("cost", unit="currencyUSD", decimals=4)]
    panels = [
        stat(
            "Tokens consumed",
            (0, 0, 4, 4),
            sql(f"SELECT sum(prompt_tokens + output_tokens) {SPENT}"),
            description="Everything the model read (prompt + image tokens) and wrote, in the selected time range.",
        ),
        stat(
            "Estimated cost",
            (4, 0, 4, 4),
            sql(f"SELECT sum(cost_usd) {SPENT}"),
            "currencyUSD",
            4,
            "Our tokens at the configured hosted model's list price (LLM_PRICE_*). The models run locally: an estimate.",
        ),
        stat(
            "Tokens saved by cache",
            (8, 0, 4, 4),
            sql(f"SELECT coalesce(sum(saved_tokens), 0) {HITS}"),
            description="Tokens of the model calls a cache hit made unnecessary.",
        ),
        stat("Cost saved by cache", (12, 0, 4, 4), sql(f"SELECT coalesce(sum(saved_cost_usd), 0) {HITS}"), "currencyUSD", 4),
        stat(
            "Understanding cache hit rate",
            (16, 0, 4, 4),
            sql(f"""SELECT count(*) FILTER (WHERE outcome = 'cache_hit')::float / nullif(count(*), 0)
                    {CALLS} AND operation = 'understand'"""),
            "percentunit",
            0,
            "Of the requests that needed understanding: how many were served from a cache (no model call).",
        ),
        stat(
            "Wasted understanding tokens",
            (20, 0, 4, 4),
            sql(f"""SELECT sum(prompt_tokens + output_tokens) FILTER (WHERE outcome IN ('rejected', 'invalid'))::float
                    / nullif(sum(prompt_tokens + output_tokens), 0) {SPENT} AND operation = 'understand'"""),
            "percentunit",
            1,
            "Share of understanding tokens spent on answers the guards threw away (rejected) or that didn't parse.",
        ),
        timeseries(
            "Tokens by operation",
            (0, 4, 12, 8),
            [
                sql(
                    f"""SELECT $__timeGroupAlias(created_at, $__interval), operation AS metric,
                     sum(prompt_tokens + output_tokens) AS tokens {SPENT} GROUP BY 1, 2 ORDER BY 1""",
                    "time_series",
                )
            ],
            stacked=True,
            bars=True,
            description="understand: one call per /ask not served from cache · caption: one call per keyframe",
        ),
        timeseries(
            "Estimated cost by operation",
            (12, 4, 12, 8),
            [
                sql(
                    f"""SELECT $__timeGroupAlias(created_at, $__interval), operation AS metric, sum(cost_usd) AS cost
                     {SPENT} GROUP BY 1, 2 ORDER BY 1""",
                    "time_series",
                )
            ],
            "currencyUSD",
            stacked=True,
            bars=True,
        ),
        table(
            "Per call",
            (0, 12, 14, 7),
            sql(f"""SELECT operation, count(*) AS calls,
                       round(avg(prompt_tokens + output_tokens)) AS "tokens per call",
                       round(avg(prompt_tokens)) AS "in", round(avg(output_tokens)) AS "out",
                       1000 * avg(cost_usd) AS "cost per 1000 calls", sum(cost_usd) AS cost,
                       avg(prompt_sec) AS "avg reading prompt",
                       round((sum(output_tokens) / nullif(sum(output_sec), 0))::numeric, 1) AS "tokens/s written",
                       max(price_reference) AS "priced at"
                    {SPENT} GROUP BY operation ORDER BY operation"""),
            overrides=per_call_columns,
        ),
        table(
            "What became of the model's answer (understand)",
            (14, 12, 10, 7),
            sql(f"""SELECT outcome, count(*) AS calls, sum(prompt_tokens + output_tokens) AS "tokens spent",
                       sum(saved_tokens) AS "tokens saved"
                    {CALLS} AND operation = 'understand' GROUP BY outcome ORDER BY calls DESC"""),
            "accepted: used as is · adjusted: used after the guards changed it · rejected: thrown away, the rules "
            "answered · invalid: reply didn't parse · cache_hit: no call made",
        ),
        table(
            "Captions per video (ingestion)",
            (0, 19, 24, 9),
            sql("""SELECT coalesce(v.title, c.video_id) AS video, count(*) AS keyframes,
                       sum(c.prompt_tokens + c.output_tokens) AS tokens,
                       round(sum(c.prompt_tokens + c.output_tokens) / nullif(max(v.duration_sec), 0) * 60) AS "tokens per minute",
                       sum(c.cost_usd) AS cost, max(c.created_at) AS "last captioned"
                    FROM llm_calls c LEFT JOIN videos v ON v.id = c.video_id
                    WHERE $__timeFilter(c.created_at) AND c.origin IN ($origin)
                      AND c.operation = 'caption' AND c.outcome <> 'cache_hit'
                    GROUP BY 1 ORDER BY tokens DESC LIMIT 50"""),
            overrides=video_columns,
        ),
    ]
    return dashboard(
        "video-rag-tokens",
        "Tokens & cost",
        "Model calls recorded in llm_calls. Cost is an estimate at a hosted model's list price.",
        panels,
        [ORIGIN],
        "now-7d",
    )


def requests_and_latency() -> dict:
    ask_total = "sum(increase(ask_requests_total[$__range]))"
    trace_link = [
        {
            "matcher": {"id": "byName", "options": "trace"},
            "properties": [
                {
                    "id": "links",
                    "value": [{"title": "Open trace", "url": f"{API_URL}/api/v1/traces/${{__value.raw}}", "targetBlank": True}],
                }
            ],
        }
    ]
    panels = [
        stat("/ask requests", (0, 0, 4, 4), prom(ask_total), decimals=0),
        stat(
            "No match",
            (4, 0, 4, 4),
            prom(f'(sum(increase(ask_requests_total{{status="no_match"}}[$__range])) or vector(0)) / {ask_total}'),
            "percentunit",
            0,
            "Requests answered with 'nothing matched'.",
        ),
        stat(
            "Rules fallback",
            (8, 0, 4, 4),
            prom(f'(sum(increase(ask_requests_total{{understood_by="rules"}}[$__range])) or vector(0)) / {ask_total}'),
            "percentunit",
            0,
            "Requests understood by the rule parser: the model failed, was disabled, or its answer was rejected by the guards.",
        ),
        stat(
            "Answer cache hits",
            (12, 0, 4, 4),
            prom(f'(sum(increase(ask_requests_total{{answer_cache="hit"}}[$__range])) or vector(0)) / {ask_total}'),
            "percentunit",
            0,
            "Whole answers served from Redis: no model call, no search.",
        ),
        stat(
            "Understanding cache hits",
            (16, 0, 4, 4),
            prom(
                'sum(increase(ask_requests_total{understanding_cache="hit"}[$__range])) / '
                'sum(increase(ask_requests_total{understanding_cache=~"hit|miss"}[$__range]))'
            ),
            "percentunit",
            0,
            "Of the requests that needed understanding (answer not cached): served without a model call.",
        ),
        stat(
            "Server errors (5xx)",
            (20, 0, 4, 4),
            prom(
                '(sum(increase(http_requests_total{status=~"5.."}[$__range])) or vector(0))'  # no 5xx yet: 0, not empty
                " / sum(increase(http_requests_total[$__range]))"
            ),
            "percentunit",
            1,
        ),
        timeseries(
            "Requests per second by endpoint",
            (0, 4, 12, 8),
            [prom("sum by (method, route) (rate(http_requests_total[$__rate_interval]))", "{{method}} {{route}}")],
            "reqps",
        ),
        timeseries(
            "Latency by endpoint (p50 dashed, p95 solid)",
            (12, 4, 12, 8),
            [
                prom(
                    "histogram_quantile(0.95, sum by (le, route) (rate(http_request_duration_seconds_bucket[$__rate_interval])))",
                    "p95 {{route}}",
                ),
                prom(
                    "histogram_quantile(0.5, sum by (le, route) (rate(http_request_duration_seconds_bucket[$__rate_interval])))",
                    "p50 {{route}}",
                    "B",
                ),
            ],
            "s",
        ),
        timeseries(
            "/ask time per stage (p95)",
            (0, 12, 12, 8),
            [
                prom(
                    "histogram_quantile(0.95, sum by (le, stage) (rate(ask_stage_seconds_bucket[$__rate_interval])))", "{{stage}}"
                )
            ],
            "s",
            "understand: model + guards (or a cache hit) · search · clips: lookup and cutting · cache: a whole answer from Redis",
        ),
        timeseries(
            "/ask results",
            (12, 12, 12, 8),
            [prom("sum by (status) (rate(ask_requests_total[$__rate_interval]))", "{{status}}")],
            "reqps",
            stacked=True,
        ),
        timeseries(
            "Chat turns by action",
            (0, 20, 12, 8),
            [prom("sum by (action) (rate(chat_turns_total[$__rate_interval]))", "{{action}}")],
            "reqps",
            "What the agent did per turn: find_clip, answer_question, find_by_image, fetch_from_pexels, list_videos, reply.",
            stacked=True,
        ),
        timeseries(
            "Chat turns: who decided",
            (12, 20, 12, 8),
            [prom("sum by (decided_by) (rate(chat_turns_total[$__rate_interval]))", "{{decided_by}}")],
            "reqps",
            "rules: a clear pattern (instant), or the model's choice was rejected by a guard · llm: the model's choice, checked.",
            stacked=True,
        ),
        table(
            "Time per step (from traces)",
            (0, 28, 12, 12),
            sql("""SELECT service, name AS step, count(*) AS count,
                       round(percentile_cont(0.5) WITHIN GROUP (ORDER BY duration_ms)::numeric) AS "p50 ms",
                       round(percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms)::numeric) AS "p95 ms",
                       round(max(duration_ms)::numeric) AS "max ms",
                       count(*) FILTER (WHERE status = 'error') AS errors
                   FROM trace_spans WHERE $__timeFilter(started_at)
                   GROUP BY service, name ORDER BY "p95 ms" DESC"""),
            "Every timed step of API requests and worker tasks (stage.* = ingestion, llm.* = model calls).",
        ),
        table(
            "Slowest requests and tasks",
            (12, 28, 12, 12),
            sql("""SELECT started_at AS time, name, service, round(duration_ms::numeric) AS "ms", status, trace_id AS trace
                   FROM trace_spans WHERE $__timeFilter(started_at) AND parent_id IS NULL
                   ORDER BY duration_ms DESC LIMIT 25"""),
            "Click a trace id to open its timeline (GET /api/v1/traces/{id}).",
            trace_link,
        ),
        table(
            "Recent errors",
            (0, 40, 24, 7),
            sql("""SELECT started_at AS time, name, service, error, trace_id AS trace
                   FROM trace_spans WHERE $__timeFilter(started_at) AND status = 'error'
                   ORDER BY started_at DESC LIMIT 25"""),
            overrides=trace_link,
        ),
    ]
    return dashboard(
        "video-rag-requests",
        "Requests & latency",
        "Live request behaviour (Prometheus) and time per step (traces in Postgres).",
        panels,
        [],
        "now-6h",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, board in {"tokens_and_cost": tokens_and_cost(), "requests_and_latency": requests_and_latency()}.items():
        (OUT / f"{name}.json").write_text(json.dumps(board, indent=2) + "\n")
        print(f"wrote {OUT / name}.json ({len(board['panels'])} panels)")


if __name__ == "__main__":
    main()
