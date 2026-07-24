import sys
import tempfile
import time
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.id_fin.detector import FINDetector


st.set_page_config(page_title="ID FIN OCR Demo", layout="wide")
st.title("Azerbaijani ID FIN OCR")
st.caption(
    "Internal demo for the ID FIN service. The production interface is FastAPI. "
    "Open **Integration Audit** in the sidebar to inspect third-party API traffic."
)

# Increment when cached detector/result objects become incompatible.
RESULT_SCHEMA_VERSION = 8


@st.cache_resource
def get_detector(use_gpu: bool) -> FINDetector:
    return FINDetector(use_gpu=use_gpu, debug=False)


def clear_detection_results() -> None:
    st.session_state.pop("detected_results", None)
    st.session_state.pop("processing_seconds", None)


if st.session_state.get("result_schema_version") != RESULT_SCHEMA_VERSION:
    clear_detection_results()
    get_detector.clear()
    st.session_state["result_schema_version"] = RESULT_SCHEMA_VERSION


use_gpu = st.sidebar.checkbox("Use GPU", value=True)
mrz_files = st.file_uploader(
    "MRZ-side images (front on older cards, back on new cards)",
    type=["png", "jpg", "jpeg", "webp", "bmp"],
    accept_multiple_files=True,
    on_change=clear_detection_results,
)

if st.button(
    "Process ID card",
    type="primary",
    disabled=not mrz_files,
):
    detected_results = []
    with tempfile.TemporaryDirectory(prefix="ocr-demo-") as directory:
        temporary_directory = Path(directory)
        started_at = time.perf_counter()
        detector = get_detector(use_gpu)
        for index, mrz_file in enumerate(mrz_files):
            mrz_path = (
                temporary_directory
                / f"mrz_{index}{Path(mrz_file.name).suffix}"
            )
            image_bytes = mrz_file.getvalue()
            mrz_path.write_bytes(image_bytes)
            detected_results.append(
                (
                    mrz_file.name,
                    image_bytes,
                    detector.detect_from_mrz(mrz_path),
                )
            )

    st.session_state["detected_results"] = detected_results
    st.session_state["processing_seconds"] = time.perf_counter() - started_at

detected_results = st.session_state.get("detected_results", [])
if detected_results:
    st.success(
        f"Completed in {st.session_state['processing_seconds']:.2f} seconds"
    )
    st.subheader("Results")
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
    for file_name, image_bytes, result in detected_results:
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
            filtered_results.append((file_name, image_bytes, result))

    st.caption(
        f"Showing {len(filtered_results)} of {len(detected_results)} results"
    )
    if not filtered_results:
        st.info("No results match the selected filters.")

    for file_name, image_bytes, result in filtered_results:
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
                st.caption(f"Confidence: {result.confidence:.2%}")

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
                else:
                    st.warning("OCR returned no MRZ details.")
