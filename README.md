# Multi-service OCR API

A FastAPI platform for exposing OCR engines as versioned third-party APIs.
The first production route extracts FIN codes from Azerbaijani ID cards.
Passport OCR is registered as a placeholder for the next service.

## Services

| Method | Route | API key | Status |
| --- | --- | --- | --- |
| `GET` | `/health` | No | Available |
| `POST` | `/v1/id-fin` | Yes | Available |
| `POST` | `/v1/id-fin/batch` | Yes | Multiple MRZ images |
| `POST` | `/v1/passport` | Yes | Returns `501` until implemented |
| `GET` | `/docs` | No | Interactive OpenAPI documentation |

Inbound `/v1/*` calls are written locally as redacted metadata. FIN values,
serial numbers, MRZ lines, and full response bodies are not retained. Raw
uploads are disabled by default; `AUDIT_STORE_PAYLOADS=true` enables bounded
retention under `AUDIT_PAYLOAD_DIR`, controlled by `AUDIT_RETENTION_DAYS` and
`AUDIT_MAX_PAYLOAD_BYTES`.

Open the Streamlit **Integration Audit** page to inspect volume, outcomes,
saved payloads, and redacted response summaries.

## Setup

Python 3.10 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Replace the example value in `.env` with a random key of at least 32
characters. Multiple unique keys may be configured as a comma-separated list:

```dotenv
API_KEYS=replace-with-a-cryptographically-random-32-plus-character-key
```

GPU acceleration is enabled by default with `USE_GPU=true`, and the project
installs `paddlepaddle-gpu`. A compatible NVIDIA driver is required. The CLI
also uses GPU by default; pass `--cpu` only for an explicit CPU run.

For local use, start the API from the repository root. One OCR engine
serializes inference on a background thread, so do not raise uvicorn
`--workers` above 1:

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

For remote integrations, terminate TLS at a trusted reverse proxy and proxy
to `127.0.0.1:8000`. Never expose port 8000 directly: API keys, identity
images, and OCR responses are sensitive.

To start both the FastAPI backend and Streamlit frontend in separate windows,
double-click `run.bat` or run:

```powershell
.\run.bat
```

The script works from any current directory, uses `.venv` when present,
creates `.env` with a random API key when missing, and installs missing
dependencies. It binds both applications to loopback. Streamlit has no remote
login and must never be exposed directly or reverse-proxied publicly.

PaddleOCR downloads its model files when the ID FIN service is used for the
first time. The detector is then reused for later requests.

## Call the ID FIN endpoint

Upload the card side containing the MRZ using the required `mrz` field. This
is the front side on older 2-line cards and the back side on new 3-line cards:

```powershell
curl.exe -X POST "https://ocr.example.internal/v1/id-fin" `
  -H "X-API-Key: your-random-32-plus-character-key" `
  -F "mrz=@C:\images\id-back.jpg"
```

Successful responses use a stable service envelope:

```json
{
  "service": "id-fin",
  "version": "v1",
  "data": {
    "fin": "7ABC123",
    "confidence": 0.96,
    "mrz_details": {
      "card_type": "new_card",
      "card_serial_number": "AA1234567"
    },
    "notes": []
  },
  "error": null
}
```

For new TD1 cards, `mrz_details.card_serial_number` contains the MRZ document
number when it matches `AA` or `AB` followed by seven digits (fixed line-1
positions 6–14). The FIN is read from optional data starting at position 16.
For older TD2 cards the numeric document number is at line-2 positions 1–9
and the FIN is at positions 29–35. It is `null` when the value cannot be
validated.

For multiple cards, repeat the `mrz` multipart field:

```powershell
curl.exe -X POST "https://ocr.example.internal/v1/id-fin/batch" `
  -H "X-API-Key: your-random-32-plus-character-key" `
  -F "mrz=@C:\images\first-back.jpg" `
  -F "mrz=@C:\images\second-back.jpg"
```

Batch responses contain `data.count` and an ordered `data.results` list. Each
result includes `index`, `file_name`, `fin`, `confidence`, `mrz_details`, and
`notes`. One PaddleOCR engine processes batch items sequentially. This keeps
GPU memory use predictable and avoids multiple model copies competing on the
same device.

Recommended `.env` values:

```dotenv
USE_GPU=true
SAVE_OCR_DEBUG_IMAGES=false
MAX_BATCH_FILES=100
```

Production is permanently pinned to PP-OCRv3 in code. Keep
`SAVE_OCR_DEBUG_IMAGES=false` unless troubleshooting locally because enabling
it writes annotated identity images to disk.

Each image is limited by `MAX_UPLOAD_BYTES` (10 MiB by default) and
`MAX_IMAGE_PIXELS` (25 megapixels by default). File signatures are verified
before OCR. A batch accepts at most `MAX_BATCH_FILES` images (100 by default)
and `MAX_BATCH_BYTES` total bytes. Uploaded working files are
removed after processing. The application does not impose a global HTTP
request or per-key request-rate limit.

`/health` reports API availability without loading GPU models. A valid
image where no FIN can be found returns `200` with `fin: null`; an OCR engine
failure returns `500` with error code `ocr_processing_failed`.

## Project structure

```text
api/                    FastAPI application, authentication, schemas, routes
services/
  id_fin/               Azerbaijani ID FIN OCR engine and service adapter
shared/                 Configuration and secure upload handling
demos/
  streamlit_app.py      Internal Streamlit demo (top-tab navigation)
  audit_dashboard.py    Integration Audit tab
  cli.py                Local command-line interface

shared/audit.py         SQLite audit store for third-party API calls
tests/                  Logic and HTTP contract tests
benchmarks/             PII-safe local accuracy and GPU performance harness
```

The HTTP route handles transport and validation. Each package under
`services/` owns its OCR implementation. `IDFinService` keeps one PaddleOCR
engine and runs inference on a single background executor thread.

## Add another OCR service

1. Create a package under `services/<service_name>/`.
2. Add a service adapter with an async `process` method.
3. Add its versioned router under `api/routes/`.
4. Include the router in `api/main.py` and list it in `/health`.
5. Add API contract and engine tests.

## Local tools

```powershell
python demos\cli.py --mrz C:\images\first-back.jpg C:\images\second-back.jpg --output json
streamlit run demos\streamlit_app.py --server.address 127.0.0.1
```

In Streamlit, use the top **Integration Audit** tab to inspect third-party
API traffic recorded while the FastAPI backend is running. Keep Streamlit on
`127.0.0.1`; its audit data is operationally sensitive.

## Tests

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```

The API tests replace the OCR engine with a fake service, so they do not load
PaddleOCR models or require real identity documents.

## GPU benchmark

This service remains a PaddleOCR GPU pipeline; it does not use ONNX Runtime.

Create an ignored private fixture manifest with
`python benchmarks\build_manifest.py --images C:\private\id-fin-corpus`.
The builder stores only SHA-256 hashes of expected FIN/serial values. Never
commit card images, local manifests, benchmark result files, or raw identity
values.

```powershell
python benchmarks\benchmark_id_fin.py `
  --manifest benchmarks\fixtures.local.json `
  --runs 7 `
  --baseline benchmarks\baseline.local.json `
  --min-improvement 0.20 `
  --output benchmarks\results-final.local.json
```

The PP-OCRv3 pipeline collects fast grayscale candidates before binarized
recovery. Matching FIN values can stop early; a lone or conflicting candidate
triggers deskew/full-image recovery. Selection prioritizes cross-attempt
agreement, TD1/TD2 checksum evidence, canonical field position, candidate
quality, and OCR confidence rather than accepting the first plausible FIN.

TD1 and TD2 candidates are evaluated together rather than letting one format
block the other. Candidate scoring uses MRZ structure, line lengths, OCR
confidence, nationality position, and checksum quality; checksum is a quality
signal rather than a hard rejection because a single OCR error can invalidate
it while leaving the FIN readable. FIN values are accepted only from the
selected format's defined field. Card serial extraction remains optional.

OCR inputs are capped at 1600 pixels and Paddle retains the measured
960-pixel detection-side limit. See `benchmarks/README.md` for methodology,
private-fixture rules, method counts, and limitations.
