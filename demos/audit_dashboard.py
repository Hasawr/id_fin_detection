"""Third-party integration audit dashboard for OCR API calls."""

from __future__ import annotations

import html
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.audit import (
    AuditEvent,
    AuditStore,
    fingerprint_api_key,
    label_api_key,
    ocr_outcome_counts,
)
from shared.config import get_settings


AUDIT_SCHEMA_VERSION = 2
PAGE_SIZE = 8

@st.cache_resource
def load_store(_schema_version: int) -> AuditStore:
    audit_settings = get_settings()
    return AuditStore(
        audit_settings.audit_db_path,
        payload_dir=audit_settings.audit_payload_dir,
    )


def client_label(fingerprint: str | None, labels: dict[str, str]) -> str:
    if not fingerprint:
        return "anonymous"
    return labels.get(fingerprint, fingerprint)


def format_when(value: str) -> str:
    return value.replace("T", " ").replace("+00:00", " UTC")


def format_when_short(value: str) -> str:
    cleaned = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(cleaned)
        return dt.strftime("%d %b · %H:%M")
    except ValueError:
        return format_when(value)[:16]


def resolve_saved_path(saved_path: str, store: AuditStore) -> Path:
    path = Path(saved_path)
    if path.is_absolute():
        return path
    return store.payload_dir.parent / path


def result_badge(event: AuditEvent) -> str:
    if not event.success:
        return event.error_code or "Request failed"
    detected, not_found = ocr_outcome_counts(event.response_body)
    if detected and not_found:
        return "Partial"
    if detected:
        return "Detected"
    if not_found:
        return "Not found"
    return "HTTP OK"


def pill_class(badge: str) -> str:
    if badge == "Detected":
        return "pill-ok"
    if badge in {"Not found", "Partial"}:
        return "pill-warn"
    if badge == "HTTP OK":
        return "pill-info"
    return "pill-err"


def pill_html(badge: str) -> str:
    safe = html.escape(badge)
    return f'<span class="pill {pill_class(badge)}">{safe}</span>'


def extracted_fields(event: AuditEvent) -> tuple[str, str]:
    body = event.response_body
    if not isinstance(body, dict):
        return "—", "—"
    data = body.get("data")
    if not isinstance(data, dict):
        return "—", "—"

    if "fin" in data:
        fin = data.get("fin") or "Not found"
        details = data.get("mrz_details")
        serial = None
        if isinstance(details, dict):
            serial = details.get("card_serial_number")
        return str(fin), str(serial) if serial else "—"

    results = data.get("results")
    if isinstance(results, list) and results:
        fins = []
        serials = []
        for item in results:
            if not isinstance(item, dict):
                continue
            fins.append(str(item.get("fin") or "Not found"))
            details = item.get("mrz_details")
            if isinstance(details, dict) and details.get("card_serial_number"):
                serials.append(str(details["card_serial_number"]))
            else:
                serials.append("—")
        return ", ".join(fins), ", ".join(serials)

    return "—", "—"


def ensure_page_in_bounds(total: int, page_size: int = PAGE_SIZE) -> int:
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = int(st.session_state.get("audit_list_page", 0))
    page = max(0, min(page, total_pages - 1))
    st.session_state["audit_list_page"] = page
    return page


def render_kpis(summary: dict) -> None:
    avg_ms = summary.get("avg_latency_ms") or 0
    st.markdown(
        f"""
        <div class="audit-kpis">
          <div class="audit-kpi"><div class="label">Calls</div><div class="value">{summary['total']}</div></div>
          <div class="audit-kpi"><div class="label">Detected</div><div class="value">{summary['ocr_detected']}</div></div>
          <div class="audit-kpi"><div class="label">Not found</div><div class="value">{summary['ocr_not_found']}</div></div>
          <div class="audit-kpi"><div class="label">Failed</div><div class="value">{summary['failed']}</div></div>
          <div class="audit-kpi">
            <div class="label">Detection rate</div>
            <div class="value">{summary['detection_rate']:.0f}%</div>
            <div class="hint">avg {avg_ms:.0f} ms</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_breakdown(summary: dict, labels: dict[str, str]) -> None:
    chips: list[str] = []
    for row in summary.get("by_service") or []:
        name = html.escape(str(row["name"] or "unknown"))
        chips.append(
            f'<div class="breakdown-item"><div class="name">{name}</div>'
            f'<div class="meta">{row["total"]} calls · {row["ocr_detected"]} detected · '
            f'{row["failed"]} failed</div></div>'
        )
    for row in summary.get("by_key") or []:
        name = html.escape(client_label(row["fingerprint"], labels))
        chips.append(
            f'<div class="breakdown-item"><div class="name">{name}</div>'
            f'<div class="meta">{row["total"]} calls · {row["ocr_detected"]} detected · '
            f'{row["failed"]} failed</div></div>'
        )
    if chips:
        st.markdown(
            '<div class="breakdown">' + "".join(chips) + "</div>",
            unsafe_allow_html=True,
        )




def render_integration_audit() -> None:
    st.markdown(
        """
    <style>
        .block-container { padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1400px; }

        .audit-kpis {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.25rem 0 1.1rem 0;
        }
        .audit-kpi {
            background: var(--secondary-background-color);
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 14px;
            padding: 0.9rem 1rem;
        }
        .audit-kpi .label {
            font-size: 0.72rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            opacity: 0.55;
            margin-bottom: 0.35rem;
        }
        .audit-kpi .value {
            font-size: 1.55rem;
            font-weight: 650;
            line-height: 1.1;
            letter-spacing: -0.02em;
        }
        .audit-kpi .hint {
            margin-top: 0.25rem;
            font-size: 0.78rem;
            opacity: 0.5;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            border-radius: 999px;
            padding: 0.15rem 0.55rem;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.01em;
            white-space: nowrap;
        }
        .pill::before {
            content: "";
            width: 0.4rem;
            height: 0.4rem;
            border-radius: 50%;
            background: currentColor;
            opacity: 0.9;
        }
        .pill-ok { background: rgba(34, 197, 94, 0.14); color: #16a34a; }
        .pill-warn { background: rgba(234, 179, 8, 0.16); color: #ca8a04; }
        .pill-err { background: rgba(239, 68, 68, 0.14); color: #dc2626; }
        .pill-info { background: rgba(59, 130, 246, 0.14); color: #2563eb; }

        .breakdown {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 0.35rem 0 0.8rem 0;
        }
        .breakdown-item {
            background: var(--secondary-background-color);
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 12px;
            padding: 0.55rem 0.75rem;
            min-width: 9.5rem;
        }
        .breakdown-item .name { font-size: 0.82rem; font-weight: 600; }
        .breakdown-item .meta { font-size: 0.72rem; opacity: 0.55; margin-top: 0.15rem; }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 14px !important;
        }

        .detail-hero {
            background: var(--secondary-background-color);
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.75rem;
        }
        .detail-hero .title {
            font-size: 1.15rem;
            font-weight: 650;
            letter-spacing: -0.02em;
            margin-bottom: 0.35rem;
        }
        .detail-hero .sub {
            font-size: 0.85rem;
            opacity: 0.65;
        }
        .kv-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.65rem;
            margin: 0.75rem 0 0.25rem 0;
        }
        .kv {
            background: var(--secondary-background-color);
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 12px;
            padding: 0.7rem 0.8rem;
        }
        .kv .k { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.04em; opacity: 0.5; }
        .kv .v { font-size: 1.05rem; font-weight: 650; margin-top: 0.2rem; word-break: break-word; }
        .meta-row {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.65rem;
            margin: 0.5rem 0 0.9rem 0;
        }
        .meta-cell .k { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.04em; opacity: 0.5; }
        .meta-cell .v { font-size: 0.88rem; margin-top: 0.15rem; word-break: break-word; }

        @media (max-width: 900px) {
            .audit-kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .kv-grid, .meta-row { grid-template-columns: 1fr; }
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


    store = load_store(AUDIT_SCHEMA_VERSION)
    settings = get_settings()
    labels = {
        fingerprint_api_key(api_key) or "": label_api_key(api_key, index)
        for index, api_key in enumerate(settings.api_keys)
    }

    # --- Header ---
    title_col, action_col = st.columns([4, 1], vertical_alignment="bottom")
    with title_col:
        st.markdown("## Integration audit")
        st.caption("Who called the OCR API, what they sent, and what came back.")
    with action_col:
        if st.button("Refresh", type="primary", use_container_width=True):
            load_store.clear()
            st.rerun()

    # --- Filters (top bar, not buried) ---
    f1, f2, f3, f4 = st.columns([1.3, 1.8, 1.2, 1.2])
    with f1:
        window_label = st.selectbox(
            "Time window",
            ["Last 1 hour", "Last 24 hours", "Last 7 days", "All time"],
            index=1,
        )
    with f2:
        outcome = st.segmented_control(
            "Outcome",
            options=["All", "Detected", "Not found", "Failed"],
            default="All",
            key="audit_outcome",
        ) or "All"
    with f3:
        event_limit = st.select_slider(
            "Show last",
            options=[25, 50, 100, 200, 500],
            value=100,
        )
    with f4:
        search = st.text_input(
            "Search",
            placeholder="FIN, serial, client…",
            label_visibility="visible",
        )

    hours = {
        "Last 1 hour": 1,
        "Last 24 hours": 24,
        "Last 7 days": 168,
        "All time": None,
    }[window_label]
    success_filter = False if outcome == "Failed" else None

    summary = store.summary(hours=hours)
    is_ocr_filter = outcome in {"Detected", "Not found"}
    fetch_limit = (
        min(max(event_limit * 10, 500), 2000)
        if is_ocr_filter or search.strip()
        else event_limit
    )
    events = store.recent_events(limit=fetch_limit, hours=hours, success=success_filter)

    if outcome == "Detected":
        events = [
            event
            for event in events
            if ocr_outcome_counts(event.response_body)[0] > 0
        ][:event_limit]
    elif outcome == "Not found":
        events = [
            event
            for event in events
            if ocr_outcome_counts(event.response_body)[1] > 0
        ][:event_limit]
    else:
        events = events[:event_limit]

    query = search.strip().lower()
    if query:
        filtered: list[AuditEvent] = []
        for event in events:
            haystack = " ".join(
                [
                    str(event.id),
                    event.result_summary or "",
                    client_label(event.api_key_fingerprint, labels),
                    event.path or "",
                    event.service or "",
                    event.error_code or "",
                    result_badge(event),
                ]
            ).lower()
            if query in haystack:
                filtered.append(event)
        events = filtered

    render_kpis(summary)
    render_breakdown(summary, labels)

    st.caption(
        f"Storage · `{store.db_path.name}` · payloads `{store.payload_dir.name}`"
    )

    if not events:
        st.info("No API calls match these filters yet. Try widening the time window or clearing search.")
        return

    # --- Selection + pagination state ---
    filter_key = (window_label, outcome, event_limit, query)
    if st.session_state.get("audit_filter_key") != filter_key:
        st.session_state["audit_filter_key"] = filter_key
        st.session_state["audit_list_page"] = 0

    total = len(events)
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page = ensure_page_in_bounds(total)
    page_events = events[page * PAGE_SIZE : (page + 1) * PAGE_SIZE]
    event_ids = {event.id for event in events}

    selected_id = st.session_state.get("audit_selected_id")
    if selected_id not in event_ids:
        selected_id = events[0].id
        st.session_state["audit_selected_id"] = selected_id

    selected: AuditEvent | None = store.get_event(selected_id)
    if selected is None:
        st.warning("Selected call is no longer available.")
        return

    # --- Master / detail ---
    list_col, detail_col = st.columns([1.05, 1.35], gap="large")

    with list_col:
        start = page * PAGE_SIZE + 1
        end = min((page + 1) * PAGE_SIZE, total)
        head_l, head_r = st.columns([2.2, 1.8])
        with head_l:
            st.markdown("#### Calls")
            st.caption(f"{start}–{end} of {total}")
        with head_r:
            p1, p2, p3 = st.columns([1, 1.4, 1])
            with p1:
                if st.button("←", disabled=page <= 0, use_container_width=True, key="audit_page_prev"):
                    st.session_state["audit_list_page"] = page - 1
                    st.rerun()
            with p2:
                st.markdown(
                    f"<div style='text-align:center;padding-top:0.4rem;opacity:0.65;"
                    f"font-size:0.85rem'>{page + 1} / {total_pages}</div>",
                    unsafe_allow_html=True,
                )
            with p3:
                if st.button(
                    "→",
                    disabled=page >= total_pages - 1,
                    use_container_width=True,
                    key="audit_page_next",
                ):
                    st.session_state["audit_list_page"] = page + 1
                    st.rerun()

        for event in page_events:
            badge = result_badge(event)
            is_selected = event.id == selected_id
            summary_text = event.result_summary or "No summary"
            client = client_label(event.api_key_fingerprint, labels)
            when = format_when_short(event.created_at)
            latency = int(round(event.latency_ms))

            with st.container(border=True):
                top_l, top_r = st.columns([3.4, 1.1])
                with top_l:
                    st.markdown(
                        f"**#{event.id}**&nbsp;&nbsp;{pill_html(badge)}",
                        unsafe_allow_html=True,
                    )
                    st.caption(summary_text)
                    st.caption(f"{when} · {client} · {latency} ms")
                with top_r:
                    if st.button(
                        "Selected" if is_selected else "View",
                        key=f"audit_open_{event.id}",
                        use_container_width=True,
                        type="primary" if is_selected else "secondary",
                        disabled=is_selected,
                    ):
                        st.session_state["audit_selected_id"] = event.id
                        st.rerun()

    with detail_col:
        badge = result_badge(selected)
        fin_value, serial_value = extracted_fields(selected)
        client = client_label(selected.api_key_fingerprint, labels)

        st.markdown(
            f"""
            <div class="detail-hero">
              <div class="title">Call #{selected.id} &nbsp; {pill_html(badge)}</div>
              <div class="sub">
                HTTP {html.escape(str(selected.status_code))} ·
                {selected.latency_ms:.0f} ms ·
                {html.escape(format_when(selected.created_at))}
              </div>
            </div>
            <div class="kv-grid">
              <div class="kv"><div class="k">FIN</div><div class="v">{html.escape(fin_value)}</div></div>
              <div class="kv"><div class="k">Card serial</div><div class="v">{html.escape(serial_value)}</div></div>
            </div>
            <div class="meta-row">
              <div class="meta-cell">
                <div class="k">Endpoint</div>
                <div class="v"><code>{html.escape(selected.method)} {html.escape(selected.path)}</code></div>
              </div>
              <div class="meta-cell">
                <div class="k">Client</div>
                <div class="v">{html.escape(client)}</div>
              </div>
              <div class="meta-cell">
                <div class="k">Host</div>
                <div class="v">{html.escape(selected.client_host or "—")}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if selected.result_summary:
            st.markdown(f"**Summary** — {selected.result_summary}")

        overview_tab, request_tab, response_tab = st.tabs(
            ["Overview", "Request files", "Response JSON"]
        )

        with overview_tab:
            overview_rows = [
                ("Service", selected.service or "—", False),
                ("Files uploaded", str(len(selected.request_files)), False),
                ("User agent", selected.user_agent or "—", True),
                ("Query", selected.request_query or "—", True),
                ("Content type", selected.request_content_type or "—", True),
                ("Error code", selected.error_code or "—", False),
                ("Payload dir", selected.payload_dir or "—", True),
            ]
            for label, value, mono in overview_rows:
                rendered = f"`{value}`" if mono else value
                st.markdown(f"**{label}**  \n{rendered}")

        with request_tab:
            if not selected.request_files:
                st.info("No uploaded files were saved for this call.")
            else:
                st.caption(f"{len(selected.request_files)} saved file(s)")
                for item in selected.request_files:
                    saved = item.get("saved_path")
                    label = item.get("file_name") or item.get("field") or "upload"
                    if not saved:
                        st.caption(f"No saved path for {label}")
                        continue

                    file_path = resolve_saved_path(str(saved), store)
                    if not file_path.exists():
                        st.warning(f"Missing file: {label}")
                        continue

                    with st.container(border=True):
                        st.markdown(f"**{label}**")
                        content_type = str(item.get("content_type") or "")
                        is_image = content_type.startswith("image/") or file_path.suffix.lower() in {
                            ".png",
                            ".jpg",
                            ".jpeg",
                            ".webp",
                            ".bmp",
                        }
                        if is_image:
                            st.image(str(file_path), use_container_width=True)
                        st.download_button(
                            "Download file",
                            data=file_path.read_bytes(),
                            file_name=file_path.name,
                            key=f"download-{selected.id}-{file_path.name}",
                            use_container_width=True,
                        )

        with response_tab:
            if selected.response_body is None:
                st.info("No response body stored.")
            elif isinstance(selected.response_body, (dict, list)):
                st.json(selected.response_body)
            else:
                st.code(str(selected.response_body))
