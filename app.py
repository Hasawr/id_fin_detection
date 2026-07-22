import os
import shutil
import tempfile
import time
from pathlib import Path
import streamlit as st
import cv2
import numpy as np

# Ensure project root is in sys.path
import sys
sys.path.append(str(Path(__file__).parent))

from fin_detector.detector import FINDetector

# Setup page config
st.set_page_config(
    page_title="Azerbaijani ID FIN Detector",
    page_icon="🆔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, premium styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 0.75rem;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    .card-header {
        font-weight: 700;
        color: #334155;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 0.5rem;
    }
    .fin-box {
        font-size: 2rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: 0.1em;
        margin: 0.5rem 0;
    }
    .conf-box {
        font-size: 0.9rem;
        font-weight: 600;
    }
    .success-text {
        color: #16A34A;
    }
    .warning-text {
        color: #D97706;
    }
    .danger-text {
        color: #DC2626;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Azerbaijani ID Card FIN Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Extract and verify FIN codes from Visual Zone (front) and Machine Readable Zone (back) images</div>', unsafe_allow_html=True)

# Ensure the local documents folder exists in the project root
DOCS_DIR = Path(__file__).parent / "documents"
DOCS_DIR.mkdir(exist_ok=True)

# Initialize FINDetector once in session state
if "detector" not in st.session_state:
    with st.spinner("Initializing PaddleOCR Engine..."):
        # We default to cpu for simple user setup, let user pick gpu in sidebar
        st.session_state["detector"] = FINDetector(use_gpu=False, debug=True)

# Sidebar options
with st.sidebar:
    st.header("Settings")
    use_gpu = st.checkbox("Use GPU (requires CUDA)", value=False)
    if st.button("Re-initialize Engine"):
        with st.spinner("Re-initializing..."):
            st.session_state["detector"] = FINDetector(use_gpu=use_gpu, debug=True)
        st.success("Re-initialized!")
        
    st.markdown("---")
    st.markdown("""
    ### App Information
    This UI runs the Azerbaijani national ID card OCR pipeline to extract the 7-character FIN code.
    
    Processed debug images highlighting the extracted areas are copied directly to the `documents/` folder.
    """)

# Layout: two columns for uploaders
col_upload_viz, col_upload_mrz = st.columns(2)

with col_upload_viz:
    st.subheader("Front Side (VIZ Zone)")
    viz_file = st.file_uploader(
        "Upload VIZ front image",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        key="viz_uploader"
    )
    if viz_file:
        st.image(viz_file, caption="Uploaded Front Side Image", use_container_width=True)

with col_upload_mrz:
    st.subheader("Back Side (MRZ Zone)")
    mrz_file = st.file_uploader(
        "Upload MRZ back image",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        key="mrz_uploader"
    )
    if mrz_file:
        st.image(mrz_file, caption="Uploaded Back Side Image", use_container_width=True)

# Run button
st.markdown("<br>", unsafe_allow_html=True)
run_pipeline = st.button("Process ID Card", type="primary", disabled=not (viz_file or mrz_file))

if run_pipeline:
    detector = st.session_state["detector"]
    
    # Create a temporary directory to save files for processing
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        viz_path = None
        mrz_path = None
        
        if viz_file:
            viz_path = tmp_path / viz_file.name
            viz_path.write_bytes(viz_file.getvalue())
            
        if mrz_file:
            mrz_path = tmp_path / mrz_file.name
            mrz_path.write_bytes(mrz_file.getvalue())
            
        # Run processing
        with st.spinner("Extracting text and locating FIN..."):
            start_time = time.time()
            if viz_path and mrz_path:
                result = detector.detect_from_both(viz_path, mrz_path)
            elif viz_path:
                result = detector.detect_from_viz(viz_path)
            else:
                result = detector.detect_from_mrz(mrz_path)
            elapsed_time = time.time() - start_time
            
        st.success(f"Processing completed in {elapsed_time:.2f} seconds.")
        st.divider()
        
        # Display Structured Outputs
        st.header("Structured Extraction Results")
        
        # Create columns for front and back results
        col_res_viz, col_res_mrz = st.columns(2)
        
        with col_res_viz:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">Front Side (VIZ) Extraction</div>', unsafe_allow_html=True)
            if viz_file:
                if result.viz_fin:
                    st.markdown(f'<div class="fin-box">{result.viz_fin}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="conf-box success-text">Confidence: {result.viz_confidence:.4f}</div>', unsafe_allow_html=True)
                    st.caption(f"Method used: {result.viz_result.method}")
                else:
                    st.markdown('<div class="fin-box danger-text">NOT FOUND</div>', unsafe_allow_html=True)
                    st.markdown('<div class="conf-box danger-text">Confidence: 0.0000</div>', unsafe_allow_html=True)
            else:
                st.info("Front image not uploaded.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_res_mrz:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">Back Side (MRZ) Extraction</div>', unsafe_allow_html=True)
            if mrz_file:
                if result.mrz_fin:
                    st.markdown(f'<div class="fin-box">{result.mrz_fin}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="conf-box success-text">Confidence: {result.mrz_confidence:.4f}</div>', unsafe_allow_html=True)
                    st.caption(f"Method used: {result.mrz_result.method}")
                    st.markdown(f"**Line 1**: `{result.mrz_result.line1}`")
                    st.markdown(f"**Line 2**: `{result.mrz_result.line2}`")
                    st.markdown(f"**Line 3**: `{result.mrz_result.line3}`")
                    checksum_class = "success-text" if result.mrz_result.checksum_valid else "warning-text"
                    checksum_status = "VALID" if result.mrz_result.checksum_valid else "INVALID / UNKNOWN"
                    st.markdown(f"**Document Checksum**: <span class='{checksum_class}'>**{checksum_status}**</span>", unsafe_allow_html=True)
                else:
                    st.markdown('<div class="fin-box danger-text">NOT FOUND</div>', unsafe_allow_html=True)
                    st.markdown('<div class="conf-box danger-text">Confidence: 0.0000</div>', unsafe_allow_html=True)
            else:
                st.info("Back image not uploaded.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Display debug images and copy them to documents folder
        st.header("Visual Debug & Bounding Boxes")
        st.info("Highlights show the detected labels and extraction regions. Images are saved to the local `documents/` folder.")
        
        col_img_viz, col_img_mrz = st.columns(2)
        
        # We need to find the files in ./debug_output/ and display them
        debug_output_dir = Path("debug_output")
        
        with col_img_viz:
            if viz_file and result.viz_fin:
                debug_viz_name = f"debug_viz_{viz_file.name}"
                debug_viz_path = debug_output_dir / debug_viz_name
                if debug_viz_path.exists():
                    st.image(str(debug_viz_path), caption="VIZ Bounding Box Highlight", use_container_width=True)
                    
                    # Copy to documents folder
                    dest_path = DOCS_DIR / debug_viz_name
                    shutil.copy(str(debug_viz_path), str(dest_path))
                    st.success(f"Uploaded and saved front image to: [documents/{debug_viz_name}](file:///{dest_path.as_posix()})")
                else:
                    st.warning("Debug VIZ image file not found.")
            elif viz_file:
                st.warning("No VIZ FIN detected, debug image not created.")
                
        with col_img_mrz:
            if mrz_file:
                debug_mrz_name = f"debug_mrz_{mrz_file.name}"
                debug_mrz_path = debug_output_dir / debug_mrz_name
                if debug_mrz_path.exists():
                    st.image(str(debug_mrz_path), caption="MRZ Text Overlay Highlight", use_container_width=True)
                    
                    # Copy to documents folder
                    dest_path = DOCS_DIR / debug_mrz_name
                    shutil.copy(str(debug_mrz_path), str(dest_path))
                    st.success(f"Uploaded and saved back image to: [documents/{debug_mrz_name}](file:///{dest_path.as_posix()})")
                else:
                    st.warning("Debug MRZ image file not found.")
            elif mrz_file:
                st.warning("No MRZ detected, debug image not created.")
                
        # Notes
        if result.notes:
            st.divider()
            with st.expander("Show System Processing Notes", expanded=False):
                for note in result.notes:
                    st.write(f"- {note}")
