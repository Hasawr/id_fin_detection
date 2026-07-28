#!/usr/bin/env bash
# Manual / debug start for the Linux GPU API.
# Prefer systemd (deploy/ocr-api.service) for stable production.
#
#   bash scripts/start_api_linux.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV_DIR="${ROOT}/.venv-gpu"
PYTHON="${VENV_DIR}/bin/python"
HOST="${OCR_BIND_HOST:-127.0.0.1}"
PORT="${OCR_BIND_PORT:-8010}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing ${PYTHON}"
  echo "Create .venv-gpu and install requirements-gpu-linux-5090.txt first."
  exit 1
fi

if [[ ! -f "${ROOT}/.env" ]]; then
  echo "Missing .env — copy .env.production.example to .env and set API_KEYS."
  exit 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "Using Python: $(which python)"
python - <<'PY'
import paddle
from services.id_fin.detector import configure_nvidia_dll_directories

configure_nvidia_dll_directories()
print("paddle", paddle.__version__, "cuda", paddle.is_compiled_with_cuda())
if not paddle.is_compiled_with_cuda():
    raise SystemExit("paddlepaddle-gpu with CUDA is required in .venv-gpu")
PY

echo "Starting API on http://${HOST}:${PORT} (uvicorn workers=1)"
exec python -m uvicorn api.main:app --host "${HOST}" --port "${PORT}"
