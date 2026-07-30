"""Public, sanitized OCR status page for third-party operators."""

from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, Response

from shared.audit import AuditEvent, AuditStore, ocr_outcome_counts


router = APIRouter(tags=["status"])

REFRESH_SECONDS = 15
RECENT_LIMIT = 25


def public_outcome(event: AuditEvent) -> str:
    """Map an audit event to a coarse, non-sensitive public label."""
    if not event.success:
        return "Rejected"
    detected, not_found = ocr_outcome_counts(event.response_body)
    if detected and not_found:
        return "Partial"
    if detected:
        return "Detected"
    if not_found:
        return "FIN not found"
    return "OK"


def public_recent_rows(events: list[AuditEvent]) -> list[dict[str, Any]]:
    """Allowlist-only fields for the public feed.

    Intentionally omits fingerprints, client hosts, user agents, filenames,
    query strings, error reasons, result summaries, and response bodies —
    those may contain PII when AUDIT_STORE_PII is enabled.
    """
    rows: list[dict[str, Any]] = []
    for event in events:
        rows.append(
            {
                "created_at": event.created_at,
                "endpoint": f"{event.method} {event.path}",
                "status_code": event.status_code,
                "latency_ms": round(event.latency_ms),
                "outcome": public_outcome(event),
            }
        )
    return rows


def _format_when(value: str) -> str:
    cleaned = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except ValueError:
        return value[:19]


def _metric_card(label: str, value: str, hint: str = "") -> str:
    hint_html = f'<div class="hint">{html.escape(hint)}</div>' if hint else ""
    return (
        f'<div class="card">'
        f'<div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div>'
        f"{hint_html}"
        f"</div>"
    )


def _summary_cards(title: str, summary: dict[str, Any]) -> str:
    total = int(summary.get("total") or 0)
    detected = int(summary.get("ocr_detected") or 0)
    not_found = int(summary.get("ocr_not_found") or 0)
    failed = int(summary.get("failed") or 0)
    rate = float(summary.get("detection_rate") or 0.0)
    median = float(summary.get("median_latency_ms") or 0.0)
    cards = "".join(
        [
            _metric_card("Calls", str(total)),
            _metric_card("Detected", str(detected)),
            _metric_card("FIN not found", str(not_found)),
            _metric_card("Rejected", str(failed)),
            _metric_card("Detection rate", f"{rate:.0f}%", "of images read"),
            _metric_card("Median latency", f"{median:.0f} ms"),
        ]
    )
    return (
        f'<section class="block">'
        f"<h2>{html.escape(title)}</h2>"
        f'<div class="grid">{cards}</div>'
        f"</section>"
    )


def _recent_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        body = (
            '<tr><td colspan="5" class="empty">'
            "No OCR calls recorded in this window yet."
            "</td></tr>"
        )
    else:
        cells: list[str] = []
        for row in rows:
            outcome = str(row["outcome"])
            css = {
                "Detected": "ok",
                "FIN not found": "warn",
                "Partial": "warn",
                "Rejected": "err",
            }.get(outcome, "info")
            cells.append(
                "<tr>"
                f"<td>{html.escape(_format_when(str(row['created_at'])))}</td>"
                f"<td><code>{html.escape(str(row['endpoint']))}</code></td>"
                f"<td>{int(row['status_code'])}</td>"
                f"<td>{int(row['latency_ms'])} ms</td>"
                f'<td><span class="pill {css}">{html.escape(outcome)}</span></td>'
                "</tr>"
            )
        body = "".join(cells)

    return (
        '<section class="block">'
        "<h2>Recent outcomes</h2>"
        "<p class=\"note\">Sanitized feed only — no client identity, filenames, "
        "images, FIN, serial, or error details.</p>"
        "<div class=\"table-wrap\"><table>"
        "<thead><tr>"
        "<th>When (UTC)</th><th>Endpoint</th><th>HTTP</th>"
        "<th>Latency</th><th>Outcome</th>"
        "</tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table></div>"
        "</section>"
    )


def render_status_html(
    *,
    hour_summary: dict[str, Any],
    day_summary: dict[str, Any],
    recent_rows: list[dict[str, Any]],
    refreshed_at: datetime | None = None,
) -> str:
    now = refreshed_at or datetime.now(timezone.utc)
    stamp = now.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="{REFRESH_SECONDS}">
  <meta name="robots" content="noindex, nofollow">
  <title>OCR API status</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #f4f6f8;
      --panel: #ffffff;
      --text: #15202b;
      --muted: #5b6773;
      --line: rgba(21, 32, 43, 0.12);
      --ok: #15803d;
      --ok-bg: rgba(21, 128, 61, 0.12);
      --warn: #a16207;
      --warn-bg: rgba(161, 98, 7, 0.14);
      --err: #b91c1c;
      --err-bg: rgba(185, 28, 28, 0.12);
      --info: #1d4ed8;
      --info-bg: rgba(29, 78, 216, 0.12);
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #0f141a;
        --panel: #171e26;
        --text: #e8eef4;
        --muted: #9aa7b5;
        --line: rgba(232, 238, 244, 0.12);
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
    }}
    main {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 1.5rem 1.1rem 2.5rem;
    }}
    header {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem 1.25rem;
      align-items: baseline;
      justify-content: space-between;
      margin-bottom: 1.25rem;
    }}
    h1 {{
      margin: 0;
      font-size: 1.55rem;
      letter-spacing: -0.02em;
    }}
    h2 {{
      margin: 0 0 0.75rem 0;
      font-size: 1rem;
      letter-spacing: 0.01em;
    }}
    .alive {{
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      font-size: 0.9rem;
      font-weight: 600;
      color: var(--ok);
    }}
    .alive::before {{
      content: "";
      width: 0.55rem;
      height: 0.55rem;
      border-radius: 50%;
      background: currentColor;
    }}
    .meta {{ color: var(--muted); font-size: 0.85rem; }}
    .block {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 1rem 1.05rem;
      margin-bottom: 1rem;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 0.65rem;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 0.7rem 0.8rem;
      background: transparent;
    }}
    .card .label {{
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--muted);
      margin-bottom: 0.25rem;
    }}
    .card .value {{
      font-size: 1.25rem;
      font-weight: 650;
      letter-spacing: -0.02em;
    }}
    .card .hint {{
      margin-top: 0.15rem;
      font-size: 0.72rem;
      color: var(--muted);
    }}
    .note {{
      margin: -0.25rem 0 0.85rem 0;
      color: var(--muted);
      font-size: 0.85rem;
    }}
    .table-wrap {{ overflow-x: auto; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
    }}
    th, td {{
      text-align: left;
      padding: 0.55rem 0.45rem;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}
    th {{
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--muted);
      font-weight: 600;
    }}
    td.empty {{
      color: var(--muted);
      text-align: center;
      padding: 1.2rem 0.5rem;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.84em;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 0.12rem 0.55rem;
      font-size: 0.75rem;
      font-weight: 650;
      white-space: nowrap;
    }}
    .pill.ok {{ background: var(--ok-bg); color: var(--ok); }}
    .pill.warn {{ background: var(--warn-bg); color: var(--warn); }}
    .pill.err {{ background: var(--err-bg); color: var(--err); }}
    .pill.info {{ background: var(--info-bg); color: var(--info); }}
    @media (max-width: 900px) {{
      .grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    }}
    @media (max-width: 560px) {{
      .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>OCR API status</h1>
        <div class="alive">API online</div>
      </div>
      <div class="meta">
        Refreshed {html.escape(stamp)} · auto every {REFRESH_SECONDS}s
      </div>
    </header>
    {_summary_cards("Last hour", hour_summary)}
    {_summary_cards("Last 24 hours", day_summary)}
    {_recent_table(recent_rows)}
  </main>
</body>
</html>
"""


def _status_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-store",
        "Pragma": "no-cache",
        "Referrer-Policy": "no-referrer",
        "X-Content-Type-Options": "nosniff",
        "X-Robots-Tag": "noindex, nofollow",
        "Content-Security-Policy": (
            "default-src 'none'; "
            "style-src 'unsafe-inline'; "
            "img-src 'none'; "
            "base-uri 'none'; "
            "form-action 'none'; "
            "frame-ancestors 'none'"
        ),
    }


@router.get(
    "/status",
    include_in_schema=False,
    summary="Public sanitized OCR status page",
)
def public_status(request: Request) -> Response:
    settings = getattr(request.app.state, "settings", None)
    if settings is None or not getattr(settings, "public_status_enabled", False):
        return Response(status_code=404, headers=_status_headers())

    store: AuditStore | None = getattr(request.app.state, "audit_store", None)
    if store is None:
        return Response(status_code=503, headers=_status_headers())

    hour_summary = store.summary(hours=1)
    day_summary = store.summary(hours=24)
    events = store.recent_events(limit=RECENT_LIMIT, hours=24)
    page = render_status_html(
        hour_summary=hour_summary,
        day_summary=day_summary,
        recent_rows=public_recent_rows(events),
    )
    return HTMLResponse(content=page, headers=_status_headers())
