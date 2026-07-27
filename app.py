import os
import sys
import shutil
import tempfile
import time
from pathlib import Path
import streamlit as st
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).parent))

from fin_detector.detector import FINDetector

# Setup page config
st.set_page_config(
    page_title="Azerbaijani ID Card FIN & ID Detector",
    page_icon="🆔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
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
        font-size: 1.8rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: 0.08em;
        margin: 0.4rem 0;
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

st.markdown('<div class="main-title">Azerbaijani ID Card FIN & ID Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Extract and verify FIN codes & ID Numbers from Machine Readable Zone (MRZ)</div>', unsafe_allow_html=True)

# Ensure the local documents folder exists in the project root
DOCS_DIR = Path(__file__).parent / "documents"
DOCS_DIR.mkdir(exist_ok=True)

# Initialize FINDetector in session state with automatic update on GPU toggle
with st.sidebar:
    st.header("Engine Settings")
    use_gpu = st.checkbox("Use GPU (CUDA Enabled)", value=True)
    
    if "detector" not in st.session_state or st.session_state.get("active_gpu_setting") != use_gpu:
        with st.spinner(f"Initializing PaddleOCR Engine (use_gpu={use_gpu})..."):
            st.session_state["detector"] = FINDetector(use_gpu=use_gpu, debug=True)
            st.session_state["active_gpu_setting"] = use_gpu
        st.toast(f"Engine active: GPU={use_gpu}", icon="⚡")
        
    if st.button("Re-initialize Engine"):
        with st.spinner("Re-initializing..."):
            st.session_state["detector"] = FINDetector(use_gpu=use_gpu, debug=True)
            st.session_state["active_gpu_setting"] = use_gpu
        st.success("Re-initialized!")
        
    st.markdown("---")
    st.markdown("""
    ### App Information
    Supports both Azerbaijani card formats:
    - **Old ID Cards (TD2, 2-Line MRZ)**: ID Number format `AZE12345678`, FIN code `7-char`
    - **New Biometric Cards (TD1, 3-Line MRZ)**: ID Number format `AA1234567`, FIN code `7-char`
    
    Processed debug images are saved directly to `documents/`.
    """)

# Layout: Left column for Upload, Right column for Results
col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    st.subheader("1. Upload MRZ Image (Back Side)")
    mrz_file = st.file_uploader(
        "Drag and drop or select MRZ back image",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        key="mrz_uploader"
    )
    
    if mrz_file:
        st.image(mrz_file, caption="Uploaded MRZ Image", use_container_width=True)

with col_right:
    st.subheader("2. Extraction Result")
    
    if mrz_file:
        detector = st.session_state["detector"]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            mrz_path = tmp_path / mrz_file.name
            mrz_path.write_bytes(mrz_file.getvalue())
            
            with st.spinner("Processing MRZ image..."):
                start_time = time.time()
                result = detector.detect_from_mrz(mrz_path)
                elapsed_time = time.time() - start_time
                
            st.caption(f"⚡ Processed in {elapsed_time:.2f} seconds")
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">MRZ Extraction Output</div>', unsafe_allow_html=True)
            
            if result.mrz_fin or result.mrz_id_number:
                if result.mrz_fin:
                    st.markdown(f'<div class="fin-box">FIN: {result.mrz_fin}</div>', unsafe_allow_html=True)
                if result.mrz_id_number:
                    st.markdown(f'### ID Number: `{result.mrz_id_number}`')
                    
                st.markdown(f'<div class="conf-box success-text">Confidence: {result.mrz_confidence:.4f}</div>', unsafe_allow_html=True)
                st.caption(f"Method used: {result.mrz_result.method if result.mrz_result else 'N/A'}")
                st.markdown("---")
                
                if result.mrz_result:
                    if result.mrz_result.is_old_card:
                        st.markdown("**Card Type**: 🟡 **Old ID Card (TD2, 2-Line MRZ)**")
                    else:
                        st.markdown("**Card Type**: 🟢 **New Biometric ID Card (TD1, 3-Line MRZ)**")
                        
                    st.markdown(f"**Line 1**: `{result.mrz_result.line1}`")
                    st.markdown(f"**Line 2**: `{result.mrz_result.line2}`")
                    if result.mrz_result.line3:
                        st.markdown(f"**Line 3**: `{result.mrz_result.line3}`")
                        
                    checksum_class = "success-text" if result.mrz_result.checksum_valid else "warning-text"
                    checksum_status = "VALID" if result.mrz_result.checksum_valid else "INVALID / UNKNOWN"
                    st.markdown(f"**Document Checksum**: <span class='{checksum_class}'>**{checksum_status}**</span>", unsafe_allow_html=True)
            else:
                st.markdown('<div class="fin-box danger-text">NOT FOUND</div>', unsafe_allow_html=True)
                st.markdown('<div class="conf-box danger-text">Confidence: 0.0000</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display debug image overlay
            debug_output_dir = Path("debug_output")
            debug_mrz_name = f"debug_mrz_{mrz_file.name}"
            debug_mrz_path = debug_output_dir / debug_mrz_name
            if debug_mrz_path.exists():
                st.subheader("Visual Debug Overlay")
                st.image(str(debug_mrz_path), caption="Annotated MRZ Bounding Boxes & Text", use_container_width=True)
                
                dest_path = DOCS_DIR / debug_mrz_name
                shutil.copy(str(debug_mrz_path), str(dest_path))
                st.success(f"Saved debug artifact to: [documents/{debug_mrz_name}](file:///{dest_path.as_posix()})")
                
            if result.notes:
                with st.expander("System Processing Notes", expanded=False):
                    for note in result.notes:
                        st.write(f"- {note}")
    else:
        st.info("👈 Please drag and drop or upload an MRZ image on the left panel to extract details.")
