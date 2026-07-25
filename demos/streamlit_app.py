import asyncio
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audit_dashboard import render_integration_audit
from services.id_fin import FINDetectionOutput
from services.id_fin.detector import OCRProcessingError
from services.id_fin.service import IDFinService
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
RESULT_SCHEMA_VERSION = 10


@st.cache_resource
def get_ocr_service(use_gpu: bool) -> IDFinService:
    return IDFinService(use_gpu=use_gpu)


def clear_detection_results() -> None:
    st.session_state.pop("detected_results", None)
    st.session_state.pop("batch_stats", None)


def clear_ocr_runtime() -> None:
    clear_detection_results()
    service = st.session_state.pop("ocr_service", None)
    if service is not None:
        service.close()
    get_ocr_service.clear()


def has_processing_error(result: FINDetectionOutput) -> bool:
    return any(note.startswith("Processing error:") for note in result.notes)


async def process_jobs(
    service: IDFinService,
    jobs: list[tuple[str, bytes, Path]],
    *,
    handle_progress: Callable[[int, int], None] | None = None,
) -> list[tuple[str, bytes, FINDetectionOutput, float]]:
    async def run_job(
        file_name: str,
        image_bytes: bytes,
        image_path: Path,
    ) -> tuple[str, bytes, FINDetectionOutput, float]:
        started = time.perf_counter()
        try:
            detection = await service.process_detection(image_path=image_path)
        except OCRProcessingError as exc:
            detection = FINDetectionOutput(
                fin=None,
                confidence=0.0,
                notes=[f"Processing error: {exc}"],
            )
        elapsed = time.perf_counter() - started
        return file_name, image_bytes, detection, elapsed

    results: list[tuple[str, bytes, FINDetectionOutput, float]] = []
    for file_name, image_bytes, image_path in jobs:
        results.append(await run_job(file_name, image_bytes, image_path))
        if handle_progress is not None:
            handle_progress(len(results), len(jobs))
    return results


def build_batch_stats(
    timed_results: list[tuple[str, bytes, FINDetectionOutput, float]],
    *,
    total_seconds: float,
    initialization_seconds: float = 0.0,
) -> dict[str, object]:
    detected = sum(1 for _, _, result, _ in timed_results if result.fin)
    errors = sum(
        1
        for _, _, result, _ in timed_results
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
        "initialization_seconds": initialization_seconds,
        "throughput": throughput,
    }


def render_id_fin_demo() -> None:
    st.title("Azerbaijani ID FIN OCR")
    st.caption(
        "Internal demo for the ID FIN service. The production interface is FastAPI."
    )

    if st.session_state.get("result_schema_version") != RESULT_SCHEMA_VERSION:
        clear_ocr_runtime()
        st.session_state["result_schema_version"] = RESULT_SCHEMA_VERSION

    settings = get_settings()

    st.subheader("Runtime")
    use_gpu = st.checkbox(
        "Use GPU",
        value=settings.use_gpu,
        on_change=clear_ocr_runtime,
    )
    st.caption(
        f"One OCR engine processes images sequentially · "
        f"max batch={settings.max_batch_files}"
    )

    mrz_files = st.file_uploader(
        "MRZ-side images (front on older cards, back on new cards)",
        type=["png", "jpg", "jpeg", "bmp"],
        accept_multiple_files=True,
        on_change=clear_detection_results,
    )

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

    process_label = (
        f"Process {len(mrz_files)} ID cards"
        if mrz_files
        else "Process ID card"
    )
    if st.button(
        process_label,
        type="primary",
        disabled=not mrz_files or has_too_many_files,
    ):
        with tempfile.TemporaryDirectory(prefix="ocr-demo-") as directory:
            temporary_directory = Path(directory)
            jobs: list[tuple[str, bytes, Path]] = []
            for index, mrz_file in enumerate(mrz_files):
                mrz_path = (
                    temporary_directory
                    / f"mrz_{index}{Path(mrz_file.name).suffix}"
                )
                image_bytes = mrz_file.getvalue()
                mrz_path.write_bytes(image_bytes)
                jobs.append((mrz_file.name, image_bytes, mrz_path))

            initialization_started_at = time.perf_counter()
            service = get_ocr_service(use_gpu)
            st.session_state["ocr_service"] = service
            initialization_seconds = (
                time.perf_counter() - initialization_started_at
            )
            started_at = time.perf_counter()
            timed_results: list[
                tuple[str, bytes, FINDetectionOutput, float]
            ] = []
            progress = st.progress(0.0, text="Preparing OCR batch…")

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

            with st.spinner(f"Processing {len(jobs)} image(s)…"):
                timed_results = asyncio.run(
                    process_jobs(
                        service,
                        jobs,
                        handle_progress=handle_progress,
                    )
                )
            total_seconds = time.perf_counter() - started_at

        st.session_state["detected_results"] = [
            (file_name, image_bytes, result, elapsed)
            for file_name, image_bytes, result, elapsed in timed_results
        ]
        st.session_state["batch_stats"] = build_batch_stats(
            timed_results,
            total_seconds=total_seconds,
            initialization_seconds=initialization_seconds,
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
    st.caption(
        "OCR processing time excludes model initialization/runtime lookup "
        f"({float(batch_stats['initialization_seconds']):.2f}s)."
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
    for index, (file_name, _, result, elapsed) in enumerate(detected_results):
        mrz_result = result.mrz_result
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
    for file_name, image_bytes, result, elapsed in detected_results:
        result_card_type = (
            result.mrz_result.card_type
            if result.mrz_result is not None
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
            filtered_results.append(
                (file_name, image_bytes, result, elapsed)
            )

    st.caption(
        f"Showing {len(filtered_results)} of {len(detected_results)} results"
    )
    if not filtered_results:
        st.info("No results match the selected filters.")

    for file_name, image_bytes, result, elapsed in filtered_results:
        with st.container(border=True):
            st.subheader(file_name)
            image_column, result_column = st.columns([1, 1.2])
            with image_column:
                st.image(
                    image_bytes,
                    caption="Processed MRZ-side image",
                    use_container_width=True,
                )

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

                mrz_result = result.mrz_result
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
