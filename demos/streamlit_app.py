import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audit_dashboard import render_integration_audit
from services.id_fin import FINDetectionOutput
from services.id_fin.detector import FINDetector
from services.id_fin.service import DetectorPool
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
RESULT_SCHEMA_VERSION = 9


@st.cache_resource
def get_detector(use_gpu: bool) -> FINDetector:
    return FINDetector(use_gpu=use_gpu, debug=False)


@st.cache_resource
def get_detector_pool(use_gpu: bool, workers: int) -> DetectorPool:
    return DetectorPool(
        workers,
        lambda: FINDetector(use_gpu=use_gpu, debug=False),
    )


def clear_detection_results() -> None:
    st.session_state.pop("detected_results", None)
    st.session_state.pop("processing_seconds", None)
    st.session_state.pop("batch_stats", None)


def detect_one_with_pool(
    pool: DetectorPool,
    image_path: Path,
) -> FINDetectionOutput:
    with pool.acquire() as detector:
        return detector.detect_from_mrz(image_path)


def process_parallel(
    pool: DetectorPool,
    jobs: list[tuple[str, bytes, Path]],
) -> list[tuple[str, bytes, FINDetectionOutput, float]]:
    results: list[tuple[str, bytes, FINDetectionOutput, float] | None] = [
        None
    ] * len(jobs)

    def run_job(
        index: int,
        file_name: str,
        image_bytes: bytes,
        image_path: Path,
    ) -> tuple[int, str, bytes, FINDetectionOutput, float]:
        started = time.perf_counter()
        detection = detect_one_with_pool(pool, image_path)
        elapsed = time.perf_counter() - started
        return index, file_name, image_bytes, detection, elapsed

    with ThreadPoolExecutor(max_workers=pool.size) as executor:
        futures = [
            executor.submit(run_job, index, file_name, image_bytes, image_path)
            for index, (file_name, image_bytes, image_path) in enumerate(jobs)
        ]
        for future in as_completed(futures):
            index, file_name, image_bytes, detection, elapsed = future.result()
            results[index] = (file_name, image_bytes, detection, elapsed)

    return [item for item in results if item is not None]


def process_sequential(
    detector: FINDetector,
    jobs: list[tuple[str, bytes, Path]],
) -> list[tuple[str, bytes, FINDetectionOutput, float]]:
    results = []
    for file_name, image_bytes, image_path in jobs:
        started = time.perf_counter()
        detection = detector.detect_from_mrz(image_path)
        elapsed = time.perf_counter() - started
        results.append((file_name, image_bytes, detection, elapsed))
    return results


def build_batch_stats(
    timed_results: list[tuple[str, bytes, FINDetectionOutput, float]],
    *,
    total_seconds: float,
    parallel: bool,
    workers: int,
) -> dict[str, object]:
    per_image = [elapsed for *_, elapsed in timed_results]
    detected = sum(1 for _, _, result, _ in timed_results if result.fin)
    failed = len(timed_results) - detected
    sum_serial = sum(per_image)
    throughput = (
        len(timed_results) / total_seconds if total_seconds > 0 else 0.0
    )
    speedup = sum_serial / total_seconds if total_seconds > 0 else 1.0
    return {
        "parallel": parallel,
        "workers": workers,
        "total_images": len(timed_results),
        "detected": detected,
        "failed": failed,
        "total_seconds": total_seconds,
        "sum_per_image_seconds": sum_serial,
        "throughput": throughput,
        "speedup_vs_serial_sum": speedup,
        "avg_per_image_seconds": (
            sum_serial / len(timed_results) if timed_results else 0.0
        ),
    }


def render_id_fin_demo() -> None:
    st.title("Azerbaijani ID FIN OCR")
    st.caption(
        "Internal demo for the ID FIN service. The production interface is FastAPI."
    )

    if st.session_state.get("result_schema_version") != RESULT_SCHEMA_VERSION:
        clear_detection_results()
        get_detector.clear()
        get_detector_pool.clear()
        st.session_state["result_schema_version"] = RESULT_SCHEMA_VERSION

    settings = get_settings()
    default_workers = settings.ocr_max_concurrency

    st.subheader("Runtime")
    runtime_cols = st.columns([1, 1.2, 1])
    with runtime_cols[0]:
        use_gpu = st.checkbox("Use GPU", value=True)
    with runtime_cols[1]:
        parallel_batch = st.checkbox(
            "Parallel batch processing",
            value=True,
            help=(
                "Process uploaded images together on a GPU worker pool "
                "(same model as the FastAPI service). Turn off to run one-by-one."
            ),
            on_change=clear_detection_results,
        )
    with runtime_cols[2]:
        workers = st.number_input(
            "GPU workers",
            min_value=1,
            max_value=16,
            value=min(max(default_workers, 1), 16),
            step=1,
            disabled=not parallel_batch,
            help=(
                f"From .env OCR_MAX_CONCURRENCY={default_workers} on this machine."
            ),
            on_change=clear_detection_results,
        )
    st.caption(
        f"Service defaults: concurrency={settings.ocr_max_concurrency}, "
        f"max batch={settings.max_batch_files}"
    )

    mrz_files = st.file_uploader(
        "MRZ-side images (front on older cards, back on new cards)",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        accept_multiple_files=True,
        on_change=clear_detection_results,
    )

    if mrz_files:
        st.caption(
            f"{len(mrz_files)} file(s) selected"
            + (
                f" · parallel with {int(workers)} worker(s)"
                if parallel_batch
                else " · sequential"
            )
        )
        if len(mrz_files) > settings.max_batch_files:
            st.warning(
                f"API batch limit is {settings.max_batch_files} images. "
                "The demo can still run locally, but production "
                "`/v1/id-fin/batch` would reject this many files."
            )

    process_label = (
        f"Process {len(mrz_files)} images in parallel"
        if parallel_batch and mrz_files
        else "Process ID card"
    )
    if st.button(
        process_label,
        type="primary",
        disabled=not mrz_files,
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

            started_at = time.perf_counter()
            if parallel_batch:
                with st.spinner(
                    f"Running {len(jobs)} images on {int(workers)} GPU workers…"
                ):
                    pool = get_detector_pool(use_gpu, int(workers))
                    timed_results = process_parallel(pool, jobs)
            else:
                with st.spinner(f"Running {len(jobs)} images sequentially…"):
                    detector = get_detector(use_gpu)
                    timed_results = process_sequential(detector, jobs)
            total_seconds = time.perf_counter() - started_at

        st.session_state["detected_results"] = [
            (file_name, image_bytes, result, elapsed)
            for file_name, image_bytes, result, elapsed in timed_results
        ]
        st.session_state["processing_seconds"] = total_seconds
        st.session_state["batch_stats"] = build_batch_stats(
            timed_results,
            total_seconds=total_seconds,
            parallel=parallel_batch,
            workers=int(workers) if parallel_batch else 1,
        )

    detected_results = st.session_state.get("detected_results", [])
    batch_stats = st.session_state.get("batch_stats")
    if not (detected_results and batch_stats):
        return

    st.success(
        f"Completed {batch_stats['total_images']} image(s) in "
        f"{batch_stats['total_seconds']:.2f}s"
        + (
            f" · {batch_stats['workers']} GPU workers"
            if batch_stats["parallel"]
            else " · sequential"
        )
    )

    st.subheader("Batch summary")
    metric_cols = st.columns(5)
    metric_cols[0].metric("Images", int(batch_stats["total_images"]))
    metric_cols[1].metric("Detected", int(batch_stats["detected"]))
    metric_cols[2].metric("Not found", int(batch_stats["failed"]))
    metric_cols[3].metric(
        "Throughput",
        f"{float(batch_stats['throughput']):.2f}/s",
    )
    metric_cols[4].metric(
        "Wall time",
        f"{float(batch_stats['total_seconds']):.2f}s",
    )

    mode_cols = st.columns(3)
    mode_cols[0].write(
        f"**Mode:** "
        f"{'Parallel GPU pool' if batch_stats['parallel'] else 'Sequential'}"
    )
    mode_cols[1].write(f"**Workers:** `{batch_stats['workers']}`")
    mode_cols[2].write(
        f"**Relative speedup:** "
        f"`{float(batch_stats['speedup_vs_serial_sum']):.2f}x` "
        f"(sum of per-image times ÷ wall time)"
    )
    st.caption(
        "Speedup compares parallel wall time to the sum of each image's own "
        "OCR duration. Values near the worker count mean good multi-worker use."
    )

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
                "Status": "ok" if result.fin else "miss",
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
                if result.fin:
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
