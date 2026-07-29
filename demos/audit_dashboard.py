"""Third-party integration audit dashboard for OCR API calls."""

from __future__ import annotations

import html
import sys
from datetime import datetime, timedelta
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
PAGE_SIZE = 16
RECEIVED_IMAGE_WIDTH_PX = 340
EVENT_FETCH_LIMIT = 200
LIVE_REFRESH_SECONDS = 12

@st.cache_resource
def load_store(_schema_version: int) -> AuditStore:
    audit_settings = get_settings()
    return AuditStore(
        audit_settings.audit_db_path,
        payload_dir=audit_settings.audit_payload_dir,
    )


def format_bytes(size: object) -> str:
    try:
        value = float(size)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""
    if value < 1024:
        return f"{value:.0f} B"
    if value < 1024 * 1024:
        return f"{value / 1024:.1f} KB"
    return f"{value / (1024 * 1024):.1f} MB"


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
    data_root = store.payload_dir.parent.resolve()
    payload_root = store.payload_dir.resolve()
    resolved = (
        path.resolve()
        if path.is_absolute()
        else (data_root / path).resolve()
    )
    if resolved != payload_root and payload_root not in resolved.parents:
        raise ValueError("Saved payload path is outside the audit payload root.")
    return resolved


def call_list_summary(event: AuditEvent) -> str:
    """Prefer a real reject reason over the old generic caption."""
    if not event.success:
        body = event.response_body if isinstance(event.response_body, dict) else {}
        detail = body.get("detail") if isinstance(body, dict) else None
        for candidate in (
            detail,
            event.error_code,
            event.result_summary,
        ):
            if not candidate:
                continue
            text = str(candidate).strip()
            if text and text != "Request rejected.":
                return text
        return f"HTTP {event.status_code}"
    return event.result_summary or "No summary"


def result_badge(event: AuditEvent) -> str:
    # Keep failure labels distinct from OCR "FIN not found". Raw FastAPI
    # details like "Not Found" (404) used to reuse the same wording.
    if not event.success:
        if event.status_code in {401, 403}:
            return "Unauthorized"
        if 400 <= event.status_code < 500:
            return "Rejected"
        if event.status_code >= 500:
            return "Server error"
        return "Request failed"
    detected, not_found = ocr_outcome_counts(event.response_body)
    if detected and not_found:
        return "Partial"
    if detected:
        return "Detected"
    if not_found:
        return "FIN not found"
    return "HTTP OK"


def pill_class(badge: str) -> str:
    if badge == "Detected":
        return "pill-ok"
    if badge in {"FIN not found", "Partial"}:
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
    # The mean is reported as the secondary figure because batch calls, which
    # process many images per request, pull it far above a typical call.
    median_ms = summary.get("median_latency_ms") or 0
    avg_ms = summary.get("avg_latency_ms") or 0
    st.markdown(
        f"""
        <div class="audit-kpis">
          <div class="audit-kpi"><div class="label">Calls</div><div class="value">{summary['total']}</div></div>
          <div class="audit-kpi"><div class="label">Detected</div><div class="value">{summary['ocr_detected']}</div></div>
          <div class="audit-kpi"><div class="label">FIN not found</div><div class="value">{summary['ocr_not_found']}</div></div>
          <div class="audit-kpi"><div class="label">Rejected / failed</div><div class="value">{summary['failed']}</div></div>
          <div class="audit-kpi">
            <div class="label">Detection rate</div>
            <div class="value">{summary['detection_rate']:.0f}%</div>
            <div class="hint">of images read</div>
          </div>
          <div class="audit-kpi">
            <div class="label">Median call</div>
            <div class="value">{median_ms:.0f} ms</div>
            <div class="hint">mean {avg_ms:.0f} ms</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def event_client_key(event: AuditEvent) -> str:
    return event.api_key_fingerprint or "none"


def _toggle_breakdown_filter(key: str, value: str) -> None:
    current = st.session_state.get(key)
    st.session_state[key] = None if current == value else value
    st.session_state["audit_list_page"] = 0
    st.session_state.pop("audit_selected_id", None)


def render_breakdown(summary: dict, labels: dict[str, str]) -> None:
    by_key = summary.get("by_key") or []
    selected_client = st.session_state.get("audit_filter_client")
    # Drop any leftover service-filter state from older builds.
    st.session_state.pop("audit_filter_service", None)

    # A single client just restates the KPIs, but keep the row visible when
    # its filter is active so the user can clear it.
    if len(by_key) <= 1 and not selected_client:
        return

    st.caption("Filter by client")
    cols = st.columns(max(len(by_key), 1))
    for col, row in zip(cols, by_key):
        value = str(row["fingerprint"] or "none")
        display = (
            "anonymous"
            if value == "none"
            else client_label(value, labels)
        )
        meta = (
            f"{row['total']} calls · {row['ocr_detected']} detected · "
            f"{row['failed']} failed"
        )
        is_on = selected_client == value
        with col:
            if st.button(
                f"{display}\n{meta}",
                key=f"audit_chip_client_{value}",
                type="primary" if is_on else "secondary",
                use_container_width=True,
                help="Click to filter the call list. Click again to clear.",
            ):
                _toggle_breakdown_filter("audit_filter_client", value)
                st.rerun()

    if selected_client:
        clear_l, clear_r = st.columns([4, 1])
        with clear_l:
            label = (
                "anonymous"
                if selected_client == "none"
                else client_label(selected_client, labels)
            )
            st.caption(f"Active: client `{label}`")
        with clear_r:
            if st.button("Clear", key="audit_clear_breakdown", use_container_width=True):
                st.session_state["audit_filter_client"] = None
                st.session_state["audit_list_page"] = 0
                st.session_state.pop("audit_selected_id", None)
                st.rerun()

@st.dialog("Clear all audit data")
def render_clear_all_dialog(store: AuditStore) -> None:
    all_time_total = store.summary(hours=None)["total"]
    payload_bytes = sum(
        path.stat().st_size
        for path in store.payload_dir.rglob("*")
        if path.is_file()
    )
    st.warning(
        "This permanently deletes every recorded API call and every saved "
        "request file on disk. This cannot be undone."
    )
    st.caption(
        f"{all_time_total} call(s) recorded · "
        f"{payload_bytes / (1024 * 1024):.1f} MB of saved payloads"
    )
    confirm_text = st.text_input(
        'Type "DELETE" to confirm',
        key="audit_clear_confirm_text",
    )
    cancel_col, confirm_col = st.columns(2)
    with cancel_col:
        if st.button("Cancel", use_container_width=True):
            st.rerun()
    with confirm_col:
        if st.button(
            "Delete everything",
            type="primary",
            use_container_width=True,
            disabled=confirm_text.strip() != "DELETE",
        ):
            result = store.clear_all()
            load_store.clear()
            st.session_state.pop("audit_selected_id", None)
            st.session_state.pop("audit_list_page", None)
            st.session_state.pop("audit_clear_confirm_text", None)
            st.session_state["audit_clear_toast"] = (
                f"Cleared {result['events']} call(s) and "
                f"{result['payload_directories']} payload folder(s)."
            )
            st.rerun()




def render_integration_audit() -> None:
    st.markdown(
        """
    <style>
        /* Use the full window: the old 1400px cap left wide empty margins on
           the monitors this page is actually watched on. */
        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 2rem;
            padding-left: 1.6rem;
            padding-right: 1.6rem;
            max-width: 100%;
        }

        .audit-kpis {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 0.6rem;
            margin: 0.25rem 0 0.8rem 0;
        }
        .audit-kpi {
            background: var(--secondary-background-color);
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 12px;
            padding: 0.6rem 0.8rem;
        }
        .audit-kpi .label {
            font-size: 0.68rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            opacity: 0.55;
            margin-bottom: 0.2rem;
        }
        .audit-kpi .value {
            font-size: 1.3rem;
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

        /* Compact call rows: each call is two tight lines instead of a card,
           so a page shows roughly twice as many without scrolling. */
        .st-key-audit_call_list div[data-testid="stVerticalBlockBorderWrapper"] {
            padding: 0.3rem 0.55rem !important;
            border-radius: 10px !important;
            margin-bottom: 0.25rem;
        }
        .st-key-audit_call_list div[data-testid="stVerticalBlock"] {
            gap: 0.1rem !important;
        }
        .st-key-audit_call_list [data-testid="stMarkdownContainer"] p {
            margin-bottom: 0 !important;
        }
        .st-key-audit_call_list button {
            padding: 0.1rem 0.4rem !important;
            min-height: 1.8rem !important;
            font-size: 0.78rem !important;
        }
        .call-row {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            flex-wrap: nowrap;
        }
        .call-row .call-id { font-weight: 650; font-size: 0.85rem; }
        .call-row .call-meta {
            font-size: 0.72rem;
            opacity: 0.55;
            margin-left: auto;
            white-space: nowrap;
        }
        .call-summary {
            font-size: 0.76rem;
            opacity: 0.75;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .detail-hero {
            background: var(--secondary-background-color);
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 14px;
            padding: 0.75rem 0.9rem;
            margin-bottom: 0.55rem;
        }
        .detail-hero .title {
            font-size: 1.05rem;
            font-weight: 650;
            letter-spacing: -0.02em;
            margin-bottom: 0.25rem;
        }
        .detail-hero .sub {
            font-size: 0.85rem;
            opacity: 0.65;
        }
        .kv-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.55rem;
            margin: 0.55rem 0 0.25rem 0;
        }
        .kv {
            background: var(--secondary-background-color);
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 12px;
            padding: 0.55rem 0.7rem;
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

        /* Six KPIs in one row need ~1200px before the values start to feel
           cramped; step down rather than shrinking them indefinitely. */
        @media (max-width: 1200px) {
            .audit-kpis { grid-template-columns: repeat(3, minmax(0, 1fr)); }
        }
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

    toast_message = st.session_state.pop("audit_clear_toast", None)
    if toast_message:
        st.toast(toast_message, icon="✅")

    # --- Header ---
    title_col, live_col, refresh_col, clear_col = st.columns(
        [2.9, 0.7, 0.85, 0.95], vertical_alignment="bottom"
    )
    with title_col:
        st.markdown("## Integration audit")
        st.caption("Who called the OCR API, what they sent, and what came back.")
    with live_col:
        live = st.toggle(
            "Live",
            key="audit_live",
            help=f"Auto-refresh every {LIVE_REFRESH_SECONDS} seconds",
        )
    with refresh_col:
        if st.button("Refresh", type="primary", use_container_width=True):
            load_store.clear()
            st.rerun()
    with clear_col:
        if st.button("Clear data", use_container_width=True):
            render_clear_all_dialog(store)

    if settings.audit_store_pii:
        st.caption(
            "⚠ Sensitive audit mode is ON — new calls retain real FIN and "
            "card serial values."
        )

    run_every = timedelta(seconds=LIVE_REFRESH_SECONDS) if live else None

    @st.fragment(run_every=run_every)
    def _audit_live_body() -> None:
        if live:
            st.caption(
                f"Live mode on · auto-refresh every {LIVE_REFRESH_SECONDS}s"
            )

        # --- Filters (top bar, not buried) ---
        with st.container(border=True):
            f1, f2 = st.columns([1, 1.6])
            with f1:
                window_label = st.selectbox(
                    "Time window",
                    ["Last 1 hour", "Last 24 hours", "Last 7 days", "All time"],
                    index=1,
                    key="audit_time_window",
                )
            with f2:
                outcome = st.segmented_control(
                    "Outcome",
                    options=["All", "Detected", "FIN not found", "Rejected"],
                    default="All",
                    # v2: old key kept stale "Not found"/"Failed" labels in session.
                    key="audit_outcome_v2",
                ) or "All"

        hours = {
            "Last 1 hour": 1,
            "Last 24 hours": 24,
            "Last 7 days": 168,
            "All time": None,
        }[window_label]
        success_filter = False if outcome == "Rejected" else None
        selected_client = st.session_state.get("audit_filter_client")

        summary = store.summary(hours=hours)
        is_ocr_filter = outcome in {"Detected", "FIN not found"}
        needs_wide_fetch = is_ocr_filter or bool(selected_client)
        fetch_limit = (
            min(max(EVENT_FETCH_LIMIT * 10, 500), 2000)
            if needs_wide_fetch
            else EVENT_FETCH_LIMIT
        )
        events = store.recent_events(limit=fetch_limit, hours=hours, success=success_filter)

        if selected_client:
            events = [
                event
                for event in events
                if event_client_key(event) == selected_client
            ]

        if outcome == "Detected":
            events = [
                event
                for event in events
                if ocr_outcome_counts(event.response_body)[0] > 0
            ][:EVENT_FETCH_LIMIT]
        elif outcome == "FIN not found":
            events = [
                event
                for event in events
                if ocr_outcome_counts(event.response_body)[1] > 0
            ][:EVENT_FETCH_LIMIT]
        else:
            events = events[:EVENT_FETCH_LIMIT]

        render_kpis(summary)
        render_breakdown(summary, labels)

        if not events:
            st.info("No API calls match these filters yet. Try widening the time window.")
            return

        # --- Selection + pagination state ---
        filter_key = (window_label, outcome, selected_client)
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
        list_col, detail_col = st.columns([1, 1.5], gap="large")

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

            with st.container(key="audit_call_list"):
                for event in page_events:
                    badge = result_badge(event)
                    is_selected = event.id == selected_id
                    summary_text = call_list_summary(event)
                    client = client_label(event.api_key_fingerprint, labels)
                    when = format_when_short(event.created_at)
                    latency = int(round(event.latency_ms))

                    with st.container(border=True):
                        row_l, row_r = st.columns([4.6, 1], vertical_alignment="center")
                        with row_l:
                            st.markdown(
                                f"<div class='call-row'>"
                                f"<span class='call-id'>#{event.id}</span>"
                                f"{pill_html(badge)}"
                                f"<span class='call-meta'>{html.escape(when)} · "
                                f"{html.escape(client)} · {latency} ms</span>"
                                f"</div>"
                                f"<div class='call-summary' title='{html.escape(summary_text)}'>"
                                f"{html.escape(summary_text)}</div>",
                                unsafe_allow_html=True,
                            )
                        with row_r:
                            if st.button(
                                "Open" if not is_selected else "Shown",
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
            reject_reason = (
                None if selected.success else call_list_summary(selected)
            )

            if reject_reason:
                detail_fields = f"""
                <div class="kv-grid" style="grid-template-columns: 1fr;">
                  <div class="kv">
                    <div class="k">Reject reason</div>
                    <div class="v">{html.escape(reject_reason)}</div>
                  </div>
                </div>
                """
            else:
                detail_fields = f"""
                <div class="kv-grid">
                  <div class="kv"><div class="k">FIN</div><div class="v">{html.escape(fin_value)}</div></div>
                  <div class="kv"><div class="k">Card serial</div><div class="v">{html.escape(serial_value)}</div></div>
                </div>
                """

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
                {detail_fields}
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

            overview_tab, request_tab, response_tab = st.tabs(
                ["Overview", "Request files", "Response JSON"]
            )

            with overview_tab:
                # A multipart content type carries a per-request boundary token
                # that is pure noise here; the media type is the useful part.
                content_type = (selected.request_content_type or "").split(";")[0]
                uploaded_bytes = sum(
                    int(item.get("size_bytes") or 0)
                    for item in selected.request_files
                )
                overview_rows = [
                    ("Reject reason", reject_reason, False),
                    ("Service", selected.service, False),
                    ("Sent by client", selected.user_agent, True),
                    (
                        "Request payload",
                        (
                            f"{len(selected.request_files)} part(s), "
                            f"{format_bytes(uploaded_bytes)}"
                            if selected.request_files
                            else ""
                        ),
                        False,
                    ),
                    ("Query", selected.request_query, True),
                    ("Content type", content_type, True),
                    ("Error code", selected.error_code, False),
                ]
                # Rows that are empty for every ordinary call add rows to scan
                # without adding information, so drop them per call.
                for label, value, mono in overview_rows:
                    if not value:
                        continue
                    st.markdown(f"**{label}**  \n{f'`{value}`' if mono else value}")

            with request_tab:
                if not selected.request_files:
                    st.info(
                        "Nothing was recorded for this request. Calls made before "
                        "request metadata was captured show no parts here."
                    )
                else:
                    uploads = [
                        item
                        for item in selected.request_files
                        if item.get("file_name")
                    ]
                    fields = [
                        item
                        for item in selected.request_files
                        if not item.get("file_name")
                    ]
                    total_bytes = sum(
                        int(item.get("size_bytes") or 0)
                        for item in selected.request_files
                    )
                    st.caption(
                        f"{len(uploads)} file(s), {len(fields)} form field(s) · "
                        f"{format_bytes(total_bytes)} received"
                    )

                    # File metadata is always recorded; the bytes only exist when
                    # AUDIT_STORE_PAYLOADS was on, so render each independently.
                    for index, item in enumerate(selected.request_files):
                        label = (
                            item.get("file_name")
                            or item.get("field")
                            or "upload"
                        )
                        with st.container(border=True):
                            meta = " · ".join(
                                part
                                for part in (
                                    f"field `{item['field']}`"
                                    if item.get("field")
                                    else "",
                                    str(item.get("content_type") or ""),
                                    format_bytes(item.get("size_bytes")),
                                )
                                if part
                            )
                            st.markdown(f"**{html.escape(str(label))}**")
                            st.caption(meta)

                            preview = item.get("value_preview")
                            if preview:
                                st.code(str(preview), language=None)

                            saved = item.get("saved_path")
                            if not saved:
                                continue
                            try:
                                file_path = resolve_saved_path(str(saved), store)
                            except ValueError:
                                st.error(f"Unsafe saved path blocked: {label}")
                                continue
                            if not file_path.exists():
                                st.warning(
                                    "The stored copy of this file has since been "
                                    "pruned."
                                )
                                continue

                            content_type = str(item.get("content_type") or "")
                            is_image = content_type.startswith(
                                "image/"
                            ) or file_path.suffix.lower() in {
                                ".png",
                                ".jpg",
                                ".jpeg",
                                ".bmp",
                            }
                            if is_image:
                                # A preview, not a full-bleed image: at container
                                # width this crowds out the rest of the call
                                # detail. Streamlit's hover control expands it.
                                st.image(
                                    str(file_path),
                                    width=RECEIVED_IMAGE_WIDTH_PX,
                                )
                            st.download_button(
                                "Download file",
                                data=file_path.read_bytes(),
                                file_name=file_path.name,
                                key=f"download-{selected.id}-{index}-{file_path.name}",
                                use_container_width=True,
                            )

                    if not any(
                        item.get("saved_path") for item in selected.request_files
                    ):
                        st.caption(
                            "File contents are not retained. Set "
                            "`AUDIT_STORE_PAYLOADS=true` to keep the uploaded "
                            "images alongside this metadata."
                        )

            with response_tab:
                if selected.response_body is None:
                    st.info("No response body stored.")
                elif isinstance(selected.response_body, (dict, list)):
                    st.json(selected.response_body)
                else:
                    st.code(str(selected.response_body))

    _audit_live_body()
