import sys
import time
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.schemas import IdFinData
from demos.api_client import (
    TimedResult,
    UploadJob,
    create_ocr_api_client,
    is_ocr_api_healthy,
    process_id_fin_jobs,
)
from demos.audit_dashboard import render_integration_audit
from shared.config import get_settings


st.set_page_config(page_title="OCR Demo", layout="wide")

# Hide the default multipage sidebar; navigation uses the top header tabs.
st.markdown(
    """
    <style>
      [data-testid="stSidebar"] { display: none; }
      [data-testid="stSidebarCollapsedControl"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Increment when cached detector/result objects become incompatible.
RESULT_SCHEMA_VERSION = 13
RESULTS_PER_PAGE = 5
MRZ_UPLOADER_KEY = "id_fin_mrz_uploader"


@st.cache_data(ttl=3, show_spinner=False)
def get_backend_health(base_url: str) -> bool:
    return is_ocr_api_healthy(base_url)


def clear_detection_results() -> None:
    st.session_state.pop("detected_results", None)
    st.session_state.pop("batch_stats", None)
    st.session_state.pop("result_page", None)


def on_mrz_files_change() -> None:
    clear_detection_results()


def has_processing_error(result: IdFinData) -> bool:
    return any(note.startswith("Processing error:") for note in result.notes)


def build_batch_stats(
    timed_results: list[TimedResult],
    *,
    total_seconds: float,
) -> dict[str, object]:
    detected = sum(1 for _, result, _ in timed_results if result.fin)
    errors = sum(
        1
        for _, result, _ in timed_results
        if has_processing_error(result)
    )
    not_found = len(timed_results) - detected - errors
    throughput = (
        len(timed_results) / total_seconds if total_seconds > 0 else 0.0
    )
    return {
        "total_images": len(timed_results),
        "detected": detected,
        "not_found": not_found,
        "errors": errors,
        "total_seconds": total_seconds,
        "throughput": throughput,
    }


def render_id_fin_demo() -> None:
    st.title("Azerbaijani ID FIN OCR")
    st.caption(
        "Internal demo for the ID FIN service. The production interface is FastAPI."
    )

    if st.session_state.get("result_schema_version") != RESULT_SCHEMA_VERSION:
        clear_detection_results()
        st.session_state["result_schema_version"] = RESULT_SCHEMA_VERSION

    settings = get_settings()

    st.subheader("Runtime")
    has_api_key = bool(settings.api_keys)
    is_backend_ready = get_backend_health(settings.ocr_api_base_url)
    if is_backend_ready:
        st.success("FastAPI OCR backend is ready.")
    else:
        st.warning(
            "FastAPI OCR backend is unavailable. Start run.bat or the API."
        )
    st.caption(
        f"Backend: {settings.ocr_api_base_url} · "
        f"one OCR engine processes images sequentially · "
        f"max batch={settings.max_batch_files}"
    )
    if not has_api_key:
        st.error("API_KEYS is not configured; Streamlit cannot call the API.")

    # Stable key keeps uploads bound across st.Page reruns; session_state is
    # the source of truth because the return value can desync in navigation apps.
    st.file_uploader(
        "MRZ-side images (front on older cards, back on new cards)",
        type=["png", "jpg", "jpeg", "bmp"],
        accept_multiple_files=True,
        key=MRZ_UPLOADER_KEY,
        on_change=on_mrz_files_change,
    )
    mrz_files = list(st.session_state.get(MRZ_UPLOADER_KEY) or [])

    has_too_many_files = bool(
        mrz_files and len(mrz_files) > settings.max_batch_files
    )
    if mrz_files:
        st.caption(
            f"{len(mrz_files)} file(s) selected"
            + " · sequential"
        )
        if has_too_many_files:
            st.error(
                f"Select at most {settings.max_batch_files} images, matching "
                "the production batch limit."
            )
    else:
        st.caption("Select one or more MRZ-side images to enable processing.")

    can_process = bool(
        mrz_files
        and not has_too_many_files
        and is_backend_ready
        and has_api_key
    )
    if mrz_files and not can_process:
        if not is_backend_ready:
            st.warning("Processing is disabled until the FastAPI backend is ready.")
        elif not has_api_key:
            st.warning("Processing is disabled until API_KEYS is configured.")
        elif has_too_many_files:
            st.warning("Processing is disabled because too many files are selected.")

    process_label = (
        f"Process {len(mrz_files)} ID cards"
        if mrz_files
        else "Process ID card"
    )
    if st.button(
        process_label,
        type="primary",
        disabled=not can_process,
    ):
        jobs: list[UploadJob] = [
            (
                mrz_file.name,
                mrz_file.getvalue(),
                mrz_file.type or "application/octet-stream",
            )
            for mrz_file in mrz_files
        ]
        started_at = time.perf_counter()
        progress = st.progress(0.0, text="Sending images to OCR API…")

        def handle_progress(completed: int, total: int) -> None:
            elapsed_so_far = time.perf_counter() - started_at
            throughput = (
                completed / elapsed_so_far
                if elapsed_so_far > 0
                else 0.0
            )
            progress.progress(
                completed / total,
                text=(
                    f"Completed {completed}/{total} "
                    f"· {throughput:.2f} images/s"
                ),
            )

        with st.status(
            f"Processing {len(jobs)} image(s) through FastAPI…",
            expanded=True,
        ) as processing_status:
            with create_ocr_api_client(
                settings.ocr_api_base_url,
                settings.api_keys[0],
            ) as client:
                timed_results = process_id_fin_jobs(
                    client,
                    jobs,
                    handle_progress=handle_progress,
                )
            processing_status.update(
                label=f"Completed {len(timed_results)} image(s)",
                state="complete",
                expanded=False,
            )
        total_seconds = time.perf_counter() - started_at

        st.session_state["detected_results"] = timed_results
        st.session_state["batch_stats"] = build_batch_stats(
            timed_results,
            total_seconds=total_seconds,
        )

    detected_results = st.session_state.get("detected_results", [])
    batch_stats = st.session_state.get("batch_stats")
    if not (detected_results and batch_stats):
        return

    st.success(
        f"Completed {batch_stats['total_images']} image(s) in "
        f"{batch_stats['total_seconds']:.2f}s"
        + " · sequential"
    )

    st.subheader("Batch summary")
    metric_cols = st.columns(6)
    metric_cols[0].metric("Images", int(batch_stats["total_images"]))
    metric_cols[1].metric("Detected", int(batch_stats["detected"]))
    metric_cols[2].metric("Not found", int(batch_stats["not_found"]))
    metric_cols[3].metric("Errors", int(batch_stats["errors"]))
    metric_cols[4].metric(
        "Throughput",
        f"{float(batch_stats['throughput']):.2f}/s",
    )
    metric_cols[5].metric(
        "Wall time",
        f"{float(batch_stats['total_seconds']):.2f}s",
    )

    st.write("**Mode:** Sequential single-engine OCR")

    summary_rows = []
    for index, (file_name, result, elapsed) in enumerate(detected_results):
        mrz_result = result.mrz_details
        summary_rows.append(
            {
                "#": index + 1,
                "File": file_name,
                "FIN": result.fin or "—",
                "Serial": (
                    mrz_result.card_serial_number
                    if mrz_result and mrz_result.card_serial_number
                    else "—"
                ),
                "Card": (
                    mrz_result.card_type
                    if mrz_result is not None
                    else "unknown"
                ),
                "Method": mrz_result.method if mrz_result else "not_found",
                "Confidence": round(float(result.confidence), 4),
                "Seconds": round(elapsed, 3),
                "Status": (
                    "error"
                    if has_processing_error(result)
                    else ("ok" if result.fin else "miss")
                ),
            }
        )
    st.dataframe(
        summary_rows,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Detailed results")
    card_filter_column, status_filter_column = st.columns(2)
    with card_filter_column:
        card_filter = st.segmented_control(
            "Card type",
            options=["All", "New", "Older", "Unknown"],
            default="All",
            key="card_type_filter",
        ) or "All"
    with status_filter_column:
        status_filter = st.segmented_control(
            "Detection status",
            options=["All", "Detected", "Not detected"],
            default="All",
            key="detection_status_filter",
        ) or "All"

    card_type_filters = {
        "New": "new_card",
        "Older": "older_card",
        "Unknown": "unknown",
    }
    filtered_results = []
    for index, (file_name, result, elapsed) in enumerate(detected_results):
        result_card_type = (
            result.mrz_details.card_type
            if result.mrz_details is not None
            else "unknown"
        )
        matches_card_type = (
            card_filter == "All"
            or result_card_type == card_type_filters.get(card_filter)
        )
        matches_status = (
            status_filter == "All"
            or (status_filter == "Detected" and result.fin is not None)
            or (status_filter == "Not detected" and result.fin is None)
        )
        if matches_card_type and matches_status:
            filtered_results.append((index, file_name, result, elapsed))

    st.caption(
        f"Showing {len(filtered_results)} of {len(detected_results)} results"
    )
    if not filtered_results:
        st.info("No results match the selected filters.")
        return

    total_pages = (
        len(filtered_results) + RESULTS_PER_PAGE - 1
    ) // RESULTS_PER_PAGE
    available_pages = list(range(1, total_pages + 1))
    if st.session_state.get("result_page") not in available_pages:
        st.session_state["result_page"] = 1
    selected_page = st.selectbox(
        "Result page",
        options=available_pages,
        key="result_page",
        format_func=lambda page: f"Page {page} of {total_pages}",
    )
    page_start = (selected_page - 1) * RESULTS_PER_PAGE
    page_results = filtered_results[
        page_start : page_start + RESULTS_PER_PAGE
    ]

    for original_index, file_name, result, elapsed in page_results:
        with st.container(border=True):
            st.subheader(file_name)
            image_column, result_column = st.columns([1, 1.2])
            with image_column:
                if (
                    original_index < len(mrz_files)
                    and mrz_files[original_index].name == file_name
                ):
                    st.image(
                        mrz_files[original_index],
                        caption="Processed MRZ-side image",
                        use_container_width=True,
                    )
                else:
                    st.info("Re-upload this image to display its preview.")

            with result_column:
                if has_processing_error(result):
                    st.error(result.notes[-1])
                elif result.fin:
                    st.success("MRZ and FIN detected")
                else:
                    failure_message = (
                        result.notes[-1]
                        if result.notes
                        else "MRZ or FIN was not detected."
                    )
                    st.error(failure_message)

                st.metric("FIN", result.fin or "Not found")
                st.caption(
                    f"Confidence: {result.confidence:.2%} · "
                    f"This image: {elapsed:.2f}s"
                )

                mrz_result = result.mrz_details
                if mrz_result is not None:
                    if mrz_result.card_type == "new_card":
                        st.info("Card type: New card (3-line MRZ)")
                    elif mrz_result.card_type == "older_card":
                        st.info("Card type: Older card (2-line MRZ)")
                    else:
                        st.warning("Card type: Unknown")

                    card_serial_number = mrz_result.card_serial_number
                    if card_serial_number:
                        st.metric(
                            "Card serial number",
                            card_serial_number,
                        )

                    st.write(f"**Detection method:** `{mrz_result.method}`")
                    if mrz_result.checksum_valid:
                        st.success("Document-number checksum is valid")
                    else:
                        st.warning(
                            "Document-number checksum is invalid or unavailable"
                        )

                    st.write("**Reconstructed MRZ lines**")
                    st.code(
                        "\n".join(
                            [
                                mrz_result.line1 or "[Line 1 not detected]",
                                mrz_result.line2 or "[Line 2 not detected]",
                                mrz_result.line3 or "[Line 3 not detected]",
                            ]
                        ),
                        language=None,
                    )
                    if result.notes:
                        with st.expander("Processing notes"):
                            for note in result.notes:
                                st.write(f"- {note}")
                else:
                    st.warning("OCR returned no MRZ details.")


id_fin_page = st.Page(
    render_id_fin_demo,
    title="ID FIN OCR",
    default=True,
)
audit_page = st.Page(
    render_integration_audit,
    title="Integration Audit",
)
st.navigation([id_fin_page, audit_page], position="top").run()
