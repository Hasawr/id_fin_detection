import csv
import html
import io
import json
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
# Keep ID previews as fixed-size thumbnails (vh/% sizing breaks on browser zoom).
st.markdown(
    """
    <style>
      :root {
        --ocr-success-bg: #e8f6ee;
        --ocr-success-fg: #146c2e;
        --ocr-error-bg: #fdecea;
        --ocr-error-fg: #b3261e;
        --ocr-muted: #6b7280;
        --ocr-border: rgba(49, 51, 63, 0.15);
      }
      @media (prefers-color-scheme: dark) {
        :root {
          --ocr-success-bg: rgba(34, 197, 94, 0.16);
          --ocr-success-fg: #4ade80;
          --ocr-error-bg: rgba(239, 68, 68, 0.16);
          --ocr-error-fg: #f87171;
          --ocr-muted: #9aa1ac;
          --ocr-border: rgba(250, 250, 250, 0.15);
        }
      }
      [data-testid="stSidebar"] { display: none; }
      [data-testid="stSidebarCollapsedControl"] { display: none; }
      /* Streamlit's own header toolbar is position:absolute, ~60px tall,
         and overlaps whatever sits at the very top of .block-container. */
      .block-container { padding-top: 4.5rem; padding-bottom: 2rem; }
      h1, h2, h3 { margin-top: 0.2rem; margin-bottom: 0.35rem; }
      div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stImage"] {
        max-width: 240px;
      }
      div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stImage"] img {
        width: 240px !important;
        max-width: 240px !important;
        height: auto !important;
        max-height: 300px !important;
        object-fit: contain;
      }
      .demo-status {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        line-height: 1.2;
        white-space: nowrap;
      }
      .demo-status-ok {
        background: var(--ocr-success-bg);
        color: var(--ocr-success-fg);
      }
      .demo-status-bad {
        background: var(--ocr-error-bg);
        color: var(--ocr-error-fg);
      }
      .ocr-result-header {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.5rem 1.5rem;
        padding-bottom: 0.6rem;
        margin-bottom: 0.6rem;
        border-bottom: 1px solid var(--ocr-border);
      }
      .ocr-result-header .ocr-field {
        display: flex;
        flex-direction: column;
        gap: 0.05rem;
        min-width: 4.5rem;
      }
      .ocr-result-header .ocr-field-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--ocr-muted);
      }
      .ocr-result-header .ocr-field-value {
        font-size: 1.05rem;
        font-weight: 600;
        font-variant-numeric: tabular-nums;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

RESULT_IMAGE_WIDTH_PX = 240

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


def to_csv(rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


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
    if st.session_state.get("result_schema_version") != RESULT_SCHEMA_VERSION:
        clear_detection_results()
        st.session_state["result_schema_version"] = RESULT_SCHEMA_VERSION

    settings = get_settings()
    has_api_key = bool(settings.api_keys)
    is_backend_ready = get_backend_health(settings.ocr_api_base_url)

    title_col, status_col = st.columns([3.2, 1.3], gap="small")
    with title_col:
        st.markdown("### Azerbaijani ID FIN OCR")
        st.caption(
            "Reads the machine-readable zone (MRZ) on an ID card and returns "
            "the personal FIN and card serial number."
        )
    with status_col:
        if is_backend_ready:
            st.markdown(
                '<div style="text-align:right;padding-top:0.35rem">'
                '<span class="demo-status demo-status-ok">Backend ready</span>'
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="text-align:right;padding-top:0.35rem">'
                '<span class="demo-status demo-status-bad">Backend offline</span>'
                "</div>",
                unsafe_allow_html=True,
            )
            st.caption("Start `run.bat` or the API, then refresh.")
    if not has_api_key:
        st.error("API_KEYS is not configured; Streamlit cannot call the API.")

    # Connection details matter when debugging, but they are not what a
    # reader needs first, so keep them subordinate to the description above.
    engine_count = settings.ocr_worker_count * settings.ocr_concurrent_attempts
    st.caption(
        f"`{settings.ocr_api_base_url}` · up to {settings.max_batch_files} "
        f"images per batch · {engine_count} OCR engine"
        f"{'s' if engine_count != 1 else ''}"
    )

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
        st.caption(f"{len(mrz_files)} file(s) selected")
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

    total_images = int(batch_stats["total_images"])
    detected_count = int(batch_stats["detected"])
    detection_rate = detected_count / total_images if total_images else 0.0

    st.success(
        f"Completed {total_images} image(s) in "
        f"{float(batch_stats['total_seconds']):.2f}s"
    )

    st.subheader("Batch summary")
    metric_cols = st.columns(5)
    metric_cols[0].metric(
        "Detection rate",
        f"{detection_rate:.0%}",
        help="Share of images where a FIN was successfully read.",
    )
    metric_cols[1].metric(
        "Detected",
        detected_count,
        help="Images where a FIN was found.",
    )
    metric_cols[2].metric(
        "Not found",
        int(batch_stats["not_found"]),
        help=(
            "Processed successfully, but no valid MRZ or FIN was present. "
            "Usually a blurred, cropped, or non-MRZ side image."
        ),
    )
    metric_cols[3].metric(
        "Errors",
        int(batch_stats["errors"]),
        help="Images the API could not process at all.",
    )
    metric_cols[4].metric(
        "Speed",
        f"{float(batch_stats['throughput']):.2f}/s",
        help=(
            f"Images per second across the whole batch "
            f"({float(batch_stats['total_seconds']):.2f}s wall time)."
        ),
    )

    card_type_labels = {
        "new_card": "New",
        "older_card": "Older",
        "unknown": "Unknown",
    }
    summary_rows = []
    for index, (file_name, result, elapsed) in enumerate(detected_results):
        mrz_result = result.mrz_details
        summary_rows.append(
            {
                "#": index + 1,
                "Status": (
                    "Error"
                    if has_processing_error(result)
                    else ("Detected" if result.fin else "Not found")
                ),
                "File": file_name,
                "FIN": result.fin or "—",
                "Serial": (
                    mrz_result.card_serial_number
                    if mrz_result and mrz_result.card_serial_number
                    else "—"
                ),
                "Card": card_type_labels.get(
                    mrz_result.card_type if mrz_result else "unknown",
                    "Unknown",
                ),
                # Only meaningful alongside a FIN; see the detail cards.
                "OCR confidence": (
                    round(float(result.confidence), 4) if result.fin else None
                ),
                "Seconds": round(elapsed, 3),
                "Method": mrz_result.method if mrz_result else "not_found",
            }
        )
    st.dataframe(
        summary_rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "OCR confidence": st.column_config.ProgressColumn(
                "OCR confidence",
                help=(
                    "Confidence of the recognised MRZ text, not of the FIN "
                    "itself. Blank when no FIN was read."
                ),
                min_value=0.0,
                max_value=1.0,
                format="%.1f%%",
            ),
            "Seconds": st.column_config.NumberColumn(
                "Seconds",
                help="Round trip through the OCR API for this image.",
                format="%.3f s",
            ),
            "Method": st.column_config.TextColumn(
                "Method",
                help="Which internal OCR attempt produced the result.",
            ),
        },
    )

    export_columns = st.columns([1, 1, 4])
    with export_columns[0]:
        st.download_button(
            "Download CSV",
            data=to_csv(summary_rows),
            file_name="id-fin-results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with export_columns[1]:
        st.download_button(
            "Download JSON",
            data=json.dumps(summary_rows, indent=2, ensure_ascii=False),
            file_name="id-fin-results.json",
            mime="application/json",
            use_container_width=True,
        )

    st.subheader("Detailed results")
    with st.container(border=True):
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

    total_pages = max(
        1,
        (len(filtered_results) + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE,
    )
    selected_page = int(st.session_state.get("result_page", 1))
    selected_page = max(1, min(selected_page, total_pages))
    st.session_state["result_page"] = selected_page

    page_start = (selected_page - 1) * RESULTS_PER_PAGE
    page_end = min(page_start + RESULTS_PER_PAGE, len(filtered_results))
    page_results = filtered_results[page_start:page_end]

    # Paging controls for a single page are three widgets that can do
    # nothing, so only render them once there is somewhere else to go.
    if total_pages > 1:
        nav_left, nav_center, nav_right = st.columns([1, 2.2, 1])
        with nav_left:
            if st.button(
                "← Previous",
                disabled=selected_page <= 1,
                use_container_width=True,
                key="result_page_prev",
            ):
                st.session_state["result_page"] = selected_page - 1
                st.rerun()
        with nav_center:
            st.markdown(
                f"<div style='text-align:center;padding-top:0.45rem;opacity:0.7;"
                f"font-size:0.9rem'>"
                f"Page {selected_page} of {total_pages}"
                f" · {page_start + 1}–{page_end} of {len(filtered_results)}"
                f"</div>",
                unsafe_allow_html=True,
            )
        with nav_right:
            if st.button(
                "Next →",
                disabled=selected_page >= total_pages,
                use_container_width=True,
                key="result_page_next",
            ):
                st.session_state["result_page"] = selected_page + 1
                st.rerun()

    for original_index, file_name, result, elapsed in page_results:
        with st.container(border=True):
            st.subheader(file_name)
            image_column, result_column = st.columns([1, 4], gap="medium")
            with image_column:
                if (
                    original_index < len(mrz_files)
                    and mrz_files[original_index].name == file_name
                ):
                    st.image(
                        mrz_files[original_index],
                        width=RESULT_IMAGE_WIDTH_PX,
                    )
                else:
                    st.caption("Re-upload to show preview.")

            with result_column:
                mrz_result = result.mrz_details
                is_error = has_processing_error(result)
                status_kind = "ok" if (result.fin and not is_error) else "bad"
                status_label = (
                    "Processing error"
                    if is_error
                    else ("FIN detected" if result.fin else "Not detected")
                )
                serial_value = (
                    mrz_result.card_serial_number
                    if mrz_result is not None and mrz_result.card_serial_number
                    else "—"
                )
                # `confidence` scores the OCR text, not whether the FIN is
                # right, so printing "98%" next to "Not detected" reads as a
                # contradiction. Only show it when there is a FIN to qualify.
                confidence_field = (
                    f"""
                      <div class="ocr-field">
                        <span class="ocr-field-label">OCR confidence</span>
                        <span class="ocr-field-value">
                          {result.confidence:.1%}
                        </span>
                      </div>
                    """
                    if result.fin
                    else ""
                )
                st.markdown(
                    f"""
                    <div class="ocr-result-header">
                      <span class="demo-status demo-status-{status_kind}">
                        {html.escape(status_label)}
                      </span>
                      <div class="ocr-field">
                        <span class="ocr-field-label">FIN</span>
                        <span class="ocr-field-value">
                          {html.escape(result.fin or "Not found")}
                        </span>
                      </div>
                      <div class="ocr-field">
                        <span class="ocr-field-label">Card serial</span>
                        <span class="ocr-field-value">
                          {html.escape(serial_value)}
                        </span>
                      </div>
                      {confidence_field}
                      <div class="ocr-field">
                        <span class="ocr-field-label">Time</span>
                        <span class="ocr-field-value">{elapsed:.2f}s</span>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if is_error:
                    st.error(result.notes[-1])
                elif not result.fin:
                    # Not a system failure: the image simply had no readable
                    # MRZ. Red would overstate it next to genuine errors.
                    st.warning(
                        result.notes[-1]
                        if result.notes
                        else "MRZ or FIN was not detected."
                    )

                if mrz_result is not None:
                    if mrz_result.card_type == "new_card":
                        card_label = "New card (3-line MRZ)"
                    elif mrz_result.card_type == "older_card":
                        card_label = "Older card (2-line MRZ)"
                    else:
                        card_label = "Unrecognised card layout"

                    if result.fin:
                        # The API exposes a single bool, so a false value
                        # cannot distinguish "checksum failed" from "no
                        # checksum to check". Claim only what it proves.
                        checksum_label = (
                            "checksum verified"
                            if mrz_result.checksum_valid
                            else "checksum not verified"
                        )
                        st.caption(
                            f"**{card_label}** · `{mrz_result.method}` · "
                            f"{checksum_label}"
                        )
                        # Older TD2 cards genuinely have two MRZ lines, so a
                        # "line 3 missing" placeholder invents a problem.
                        mrz_lines = [
                            mrz_result.line1 or "[Line 1 not detected]",
                            mrz_result.line2 or "[Line 2 not detected]",
                        ]
                        if mrz_result.card_type != "older_card":
                            mrz_lines.append(
                                mrz_result.line3 or "[Line 3 not detected]"
                            )
                        st.caption("Machine-readable zone")
                        st.code("\n".join(mrz_lines), language=None)
                    else:
                        # With no MRZ recognised there is no layout to name
                        # and no checksum to report; only the raw text the
                        # OCR did see is useful for diagnosing the image.
                        recognised = [
                            line
                            for line in (
                                mrz_result.line1,
                                mrz_result.line2,
                                mrz_result.line3,
                            )
                            if line
                        ]
                        if recognised:
                            st.caption("Text the OCR did read")
                            st.code("\n".join(recognised), language=None)
                        else:
                            st.caption("No text was recognised in this image.")

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
