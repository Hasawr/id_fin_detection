#!/usr/bin/env bash
# Validate Linux GPU OCR prerequisites for RTX 50-series hosts.
# Run from the repository root with the GPU venv activated:
#   source .venv-gpu/bin/activate
#   bash scripts/validate_gpu_linux.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== Python =="
python -V

echo "== NVIDIA driver =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. Install an NVIDIA driver that supports CUDA 12.9+."
  exit 1
fi
nvidia-smi

echo "== Configure pip NVIDIA lib path (same as FINDetector) =="
python - <<'PY'
from services.id_fin.detector import configure_nvidia_dll_directories
import os

configure_nvidia_dll_directories()
print("LD_LIBRARY_PATH=", os.environ.get("LD_LIBRARY_PATH", ""))
PY

echo "== Paddle CUDA build =="
python - <<'PY'
import paddle

print("paddle", paddle.__version__)
print("compiled_with_cuda", paddle.is_compiled_with_cuda())
if not paddle.is_compiled_with_cuda():
    raise SystemExit(
        "paddlepaddle-gpu with CUDA is required. "
        "Install paddlepaddle-gpu==3.2.1 from the cu129 index."
    )
PY

echo "== Smoke PaddleOCR PP-OCRv3 GPU init =="
python - <<'PY'
from services.id_fin.detector import configure_nvidia_dll_directories, PRODUCTION_OCR_VERSION
from paddleocr import PaddleOCR

configure_nvidia_dll_directories()
ocr = PaddleOCR(
    use_angle_cls=False,
    lang="en",
    show_log=False,
    use_gpu=True,
    max_text_length=40,
    det_limit_side_len=736,
    det_limit_type="max",
    ocr_version=PRODUCTION_OCR_VERSION,
)
print("PaddleOCR ready:", PRODUCTION_OCR_VERSION, type(ocr).__name__)
PY

echo "GPU stack looks ready. Start the API with USE_GPU=true and test one MRZ image."
