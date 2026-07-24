"""Third-party integration audit dashboard for OCR API calls."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.audit import AuditEvent, AuditStore, fingerprint_api_key, get_audit_store, label_api_key
from shared.config import get_settings


st.set_page_config(page_title="Integration Audit", layout="wide")
st.title("Integration audit")
st.caption("Review third-party OCR API calls: who called, what they sent, and what came back.")


@st.cache_resource
def load_store() -> AuditStore:
    return get_audit_store()


def client_label(fingerprint: str | None, labels: dict[str, str]) -> str:
    if not fingerprint:
        return "none"
    return labels.get(fingerprint, fingerprint)


def format_when(value: str) -> str:
    return value.replace("T", " ").replace("+00:00", " UTC")


def resolve_saved_path(saved_path: str, store: AuditStore) -> Path:
    path = Path(saved_path)
    if path.is_absolute():
        return path
    return store.payload_dir.parent / path


def result_badge(event: AuditEvent) -> str:
    if event.success:
        return "OK"
    return event.error_code or "Failed"


def is_fin_not_found(event: AuditEvent) -> bool:
    summary = (event.result_summary or "").lower()
    return "fin not found" in summary or "no fin found" in summary


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


store = load_store()
settings = get_settings()
labels = {
    fingerprint_api_key(api_key) or "": label_api_key(api_key, index)
    for index, api_key in enumerate(settings.api_keys)
}

with st.sidebar:
    st.header("Filters")
    window_label = st.selectbox(
        "Time window",
        ["Last 1 hour", "Last 24 hours", "Last 7 days", "All time"],
        index=1,
    )
    hours = {
        "Last 1 hour": 1,
        "Last 24 hours": 24,
        "Last 7 days": 168,
        "All time": None,
    }[window_label]
    outcome = st.radio("Outcome", ["All", "Succeeded", "Failed"], horizontal=True)
    success_filter = {"All": None, "Succeeded": True, "Failed": False}[outcome]
    fin_not_found_only = st.checkbox("FIN not found only", value=False)
    event_limit = st.slider("Show last", 25, 500, 100, 25)
    if st.button("Refresh", type="primary", use_container_width=True):
        load_store.clear()
        st.rerun()
    st.divider()
    st.caption(f"DB `{store.db_path.name}`")
    st.caption(f"Payloads `{store.payload_dir}`")

summary = store.summary(hours=hours)
fetch_limit = event_limit if not fin_not_found_only else min(max(event_limit * 10, 500), 2000)
events = store.recent_events(limit=fetch_limit, hours=hours, success=success_filter)
if fin_not_found_only:
    events = [event for event in events if is_fin_not_found(event)][:event_limit]

# --- Overview ---
k1, k2, k3, k4 = st.columns(4)
k1.metric("Calls", summary["total"])
k2.metric("Succeeded", summary["succeeded"])
k3.metric("Failed", summary["failed"])
k4.metric(
    "Success rate",
    f"{summary['success_rate']:.0f}%",
    help=f"Avg latency {summary['avg_latency_ms']:.0f} ms",
)

if summary["by_service"] or summary["by_key"]:
    left, right = st.columns(2)
    with left:
        st.markdown("##### By service")
        st.dataframe(
            [
                {
                    "Service": row["name"],
                    "Calls": row["total"],
                    "OK": row["succeeded"],
                    "Failed": row["failed"],
                }
                for row in summary["by_service"]
            ],
            hide_index=True,
            use_container_width=True,
        )
    with right:
        st.markdown("##### By client")
        st.dataframe(
            [
                {
                    "Client": client_label(row["fingerprint"], labels),
                    "Calls": row["total"],
                    "OK": row["succeeded"],
                    "Failed": row["failed"],
                }
                for row in summary["by_key"]
            ],
            hide_index=True,
            use_container_width=True,
        )

st.divider()

if not events:
    st.info("No API calls recorded for these filters yet.")
    st.stop()

# --- Call list (full width) ---
st.subheader("Recent calls")
st.caption(f"Showing {len(events)} call(s) · click a row to inspect")

table_rows = [
    {
        "ID": event.id,
        "When": format_when(event.created_at),
        "Result": result_badge(event),
        "Summary": event.result_summary or "—",
        "ms": int(round(event.latency_ms)),
        "Client": client_label(event.api_key_fingerprint, labels),
    }
    for event in events
]
selection = st.dataframe(
    table_rows,
    hide_index=True,
    use_container_width=True,
    height=min(420, 56 + 35 * len(table_rows)),
    on_select="rerun",
    selection_mode="single-row",
    key="audit_calls_table",
    column_config={
        "ID": st.column_config.NumberColumn(width="small"),
        "When": st.column_config.TextColumn(width="medium"),
        "Result": st.column_config.TextColumn(width="small"),
        "Summary": st.column_config.TextColumn(width="large"),
        "ms": st.column_config.NumberColumn("Latency", width="small"),
        "Client": st.column_config.TextColumn(width="medium"),
    },
)

selected_rows = selection.selection.rows if selection.selection else []
if selected_rows:
    selected_id = events[selected_rows[0]].id
    st.session_state["audit_selected_id"] = selected_id
else:
    selected_id = st.session_state.get("audit_selected_id", events[0].id)
    if selected_id not in {event.id for event in events}:
        selected_id = events[0].id
        st.session_state["audit_selected_id"] = selected_id

selected: AuditEvent | None = store.get_event(selected_id)
if selected is None:
    st.warning("Selected call is no longer available.")
    st.stop()

# --- Detail (below table) ---
st.divider()
st.subheader(f"Call #{selected.id}")

status_line = (
    f"**{result_badge(selected)}** · HTTP {selected.status_code} · "
    f"{selected.latency_ms:.0f} ms · {format_when(selected.created_at)}"
)
if selected.success:
    st.success(status_line)
else:
    st.error(status_line)

if selected.result_summary:
    st.markdown(f"**Result:** {selected.result_summary}")

meta1, meta2, meta3 = st.columns(3)
meta1.markdown(f"**Endpoint**  \n`{selected.method} {selected.path}`")
meta2.markdown(f"**Client**  \n{client_label(selected.api_key_fingerprint, labels)}")
meta3.markdown(f"**Host**  \n{selected.client_host or '—'}")

fin_value, serial_value = extracted_fields(selected)
value1, value2 = st.columns(2)
value1.metric("FIN", fin_value)
value2.metric("Card serial", serial_value)

overview_tab, request_tab, response_tab = st.tabs(
    ["Overview", "Request", "Response"]
)

with overview_tab:
    overview_rows = [
        ("Service", selected.service or "—"),
        ("Files uploaded", str(len(selected.request_files))),
        ("User agent", selected.user_agent or "—"),
        ("Query", selected.request_query or "—"),
        ("Content type", selected.request_content_type or "—"),
        ("Error code", selected.error_code or "—"),
    ]
    for label, value in overview_rows:
        st.markdown(f"**{label}**  \n{value}")

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
                    "Download",
                    data=file_path.read_bytes(),
                    file_name=file_path.name,
                    key=f"download-{selected.id}-{file_path.name}",
                    use_container_width=True,
                )

    if selected.payload_dir:
        st.caption(f"Payload dir: `{selected.payload_dir}`")

with response_tab:
    if selected.response_body is None:
        st.info("No response body stored.")
    elif isinstance(selected.response_body, (dict, list)):
        st.json(selected.response_body)
    else:
        st.code(str(selected.response_body))
