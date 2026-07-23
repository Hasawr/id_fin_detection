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
st.caption("Local record of every third-party API call: result, payload, and response.")


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


store = load_store()
settings = get_settings()
labels = {
    fingerprint_api_key(api_key) or "": label_api_key(api_key, index)
    for index, api_key in enumerate(settings.api_keys)
}

with st.sidebar:
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
    outcome = st.selectbox("Outcome", ["All", "Succeeded", "Failed"])
    success_filter = {"All": None, "Succeeded": True, "Failed": False}[outcome]
    event_limit = st.slider("Show last", 25, 500, 100, 25)
    if st.button("Refresh", type="primary", use_container_width=True):
        st.rerun()

summary = store.summary(hours=hours)
events = store.recent_events(limit=event_limit, hours=hours, success=success_filter)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Calls", summary["total"])
m2.metric("OK", summary["succeeded"])
m3.metric("Failed", summary["failed"])
m4.metric("Avg latency", f"{summary['avg_latency_ms']:.0f} ms")

if summary["by_service"] or summary["by_key"]:
    left, right = st.columns(2)
    with left:
        st.write("**By service**")
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
        st.write("**By client**")
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
st.subheader("Calls")

if not events:
    st.info("No API calls recorded for these filters yet.")
    st.stop()

table_rows = [
    {
        "ID": event.id,
        "When": format_when(event.created_at),
        "Service": event.service or event.path,
        "Result": "OK" if event.success else "Failed",
        "Status": event.status_code,
        "Summary": event.result_summary or event.error_code or "—",
        "Files": len(event.request_files),
        "Latency ms": event.latency_ms,
        "Client": client_label(event.api_key_fingerprint, labels),
    }
    for event in events
]
st.dataframe(table_rows, hide_index=True, use_container_width=True)

event_ids = [event.id for event in events]
selected_id = st.selectbox(
    "Inspect call",
    event_ids,
    format_func=lambda event_id: next(
        (
            f"#{event.id} · {event.service or event.path} · "
            f"{'OK' if event.success else 'Failed'} · "
            f"{event.result_summary or event.error_code or event.status_code}"
            for event in events
            if event.id == event_id
        ),
        str(event_id),
    ),
)

selected: AuditEvent | None = store.get_event(selected_id)
if selected is None:
    st.warning("Selected call is no longer available.")
    st.stop()

meta_cols = st.columns(4)
meta_cols[0].markdown(f"**Path**  \n`{selected.method} {selected.path}`")
meta_cols[1].markdown(f"**Client**  \n{client_label(selected.api_key_fingerprint, labels)}")
meta_cols[2].markdown(f"**Host**  \n{selected.client_host or '—'}")
meta_cols[3].markdown(f"**When**  \n{format_when(selected.created_at)}")

detail_left, detail_right = st.columns(2)

with detail_left:
    st.markdown("**Request (saved locally)**")
    st.write(
        {
            "content_type": selected.request_content_type,
            "query": selected.request_query,
            "user_agent": selected.user_agent,
            "payload_dir": selected.payload_dir,
            "files": selected.request_files,
        }
    )
    for item in selected.request_files:
        saved = item.get("saved_path")
        if not saved:
            continue
        file_path = resolve_saved_path(str(saved), store)
        label = item.get("file_name") or item.get("field") or file_path.name
        if not file_path.exists():
            st.caption(f"Missing file: {label}")
            continue
        content_type = str(item.get("content_type") or "")
        if content_type.startswith("image/") or file_path.suffix.lower() in {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".bmp",
        }:
            st.image(str(file_path), caption=str(label), use_container_width=True)
        st.download_button(
            f"Download {label}",
            data=file_path.read_bytes(),
            file_name=file_path.name,
            key=f"download-{selected.id}-{file_path.name}",
        )

with detail_right:
    st.markdown("**Response (saved locally)**")
    if selected.response_body is None:
        st.write("No response body stored.")
    elif isinstance(selected.response_body, (dict, list)):
        st.json(selected.response_body)
    else:
        st.code(str(selected.response_body))

st.caption(f"DB: `{store.db_path}` · payloads: `{store.payload_dir}`")
