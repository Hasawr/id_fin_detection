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
| `GET` | `/docs` | No | OpenAPI documentation |

Inbound `/v1/*` calls are written locally for audit:
- metadata and full JSON responses in `AUDIT_DB_PATH` (default `data/audit.db`)
- uploaded images and request bodies in `AUDIT_PAYLOAD_DIR`
  (default `data/audit_payloads`)

Open the Streamlit **Integration Audit** page to inspect volume, outcomes,
saved payloads, and response bodies.

## Setup

Python 3.10 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Replace the example value in `.env` with a long random key. Multiple keys may
be configured as a comma-separated list:

```dotenv
API_KEYS=first-client-key,second-client-key
```

GPU acceleration is enabled by default with `USE_GPU=true`, and the project
installs `paddlepaddle-gpu`. A compatible NVIDIA driver is required. The CLI
also uses GPU by default; pass `--cpu` only for an explicit CPU run.

Start the API from the repository root (one process — the in-app GPU pool
handles parallelism; do not raise uvicorn `--workers` above 1 on the OCR host):

```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

To start both the FastAPI backend and Streamlit frontend in separate windows,
double-click `run.bat` or run:

```powershell
.\run.bat
```

The script works from any current directory, uses `.venv` when present,
creates `.env` from the example when missing, and installs missing
dependencies.

PaddleOCR downloads its model files when the ID FIN service is used for the
first time. The detector is then reused for later requests.

## Call the ID FIN endpoint

Upload the card side containing the MRZ using the required `mrz` field. This
is the front side on older 2-line cards and the back side on new 3-line cards:

```powershell
curl.exe -X POST "http://localhost:8000/v1/id-fin" `
  -H "X-API-Key: first-client-key" `
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
curl.exe -X POST "http://localhost:8000/v1/id-fin/batch" `
  -H "X-API-Key: first-client-key" `
  -F "mrz=@C:\images\first-back.jpg" `
  -F "mrz=@C:\images\second-back.jpg"
```

Batch responses contain `data.count` and an ordered `data.results` list. Each
result includes `index`, `file_name`, `fin`, `confidence`, `mrz_details`, and
`notes`. Batch items and concurrent HTTP requests share a bounded GPU worker
pool (`OCR_MAX_CONCURRENCY`). Extra requests wait for a free worker instead of
overloading VRAM.

| Host | Suggested `OCR_MAX_CONCURRENCY` |
| --- | --- |
| Dev laptop (6–8 GB VRAM) | `1`–`2` |
| Mid GPU (12–16 GB) | `3`–`4` |
| Core PC (RTX 5090 32 GB + 14700K / 64 GB RAM) | `8` (start), `10`–`12` if VRAM headroom remains |

On the core server, set in `.env`:

```dotenv
USE_GPU=true
OCR_MAX_CONCURRENCY=8
MAX_BATCH_FILES=20
```

Each image is limited by `MAX_UPLOAD_BYTES` (10 MiB by default) and
`MAX_IMAGE_PIXELS` (25 megapixels by default). File signatures are verified
before OCR. A batch accepts at most `MAX_BATCH_FILES` images (20 on the core
PC example above). Uploaded files are stored in a request-specific temporary
directory and removed after processing.

## Project structure

```text
api/                    FastAPI application, authentication, schemas, routes
services/
  id_fin/               Azerbaijani ID FIN OCR engine and service adapter
  passport/             Passport service placeholder
shared/                 Configuration and secure upload handling
demos/
  cli.py                Local command-line interface
  streamlit_app.py      Internal Streamlit demo
  pages/                Streamlit pages (Integration Audit)
shared/audit.py         SQLite audit store for third-party API calls
tests/                  Logic and HTTP contract tests
benchmarks/             PII-safe local accuracy and GPU performance harness
```

The HTTP route handles transport and validation. Each package under
`services/` owns its OCR implementation. `IDFinService` keeps a pool of
PaddleOCR workers and runs inference on a thread-pool executor so many
requests can progress at once up to `OCR_MAX_CONCURRENCY`.

## Add another OCR service

1. Create a package under `services/<service_name>/`.
2. Add a service adapter with an async `process` method.
3. Add its versioned router under `api/routes/`.
4. Include the router in `api/main.py` and list it in `/health`.
5. Add API contract and engine tests.

## Local tools

```powershell
python demos\cli.py --mrz C:\images\first-back.jpg C:\images\second-back.jpg --output json
streamlit run demos\streamlit_app.py
```

In Streamlit, use the sidebar page **Integration Audit** to inspect third-party
API traffic recorded while the FastAPI backend is running.

## Tests

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```

The API tests replace the OCR engine with a fake service, so they do not load
PaddleOCR models or require real identity documents.

## GPU benchmark

This service remains a PaddleOCR GPU pipeline; it does not use ONNX Runtime.

Create an ignored private fixture manifest from
`benchmarks/fixtures.example.json`. Store only SHA-256 hashes of expected FIN
values and never commit card images, local manifests, benchmark result files,
or raw FIN values.

```powershell
python benchmarks\benchmark_id_fin.py `
  --manifest benchmarks\fixtures.local.json `
  --runs 7 `
  --baseline benchmarks\baseline.local.json `
  --min-improvement 0.20 `
  --output benchmarks\results-final.local.json
```

On Windows 11, Python 3.12.10, PaddlePaddle GPU 2.6.2, CUDA 11.8, cuDNN 8.6,
and an NVIDIA GeForce RTX 4060 Laptop GPU, the two-card TD1/TD2 fixture set
improved from 170.9 ms to 126.2 ms warm median latency. Throughput increased
from 5.89 to 8.05 images/s, full-image fallback fell from 50% to 0%, and all
expected FIN/card-type results remained unchanged.

The optimized pipeline localizes the card and tries lazy attempts in order:
cleaned bottom strip, detected MRZ rectangle, deskewed MRZ rectangle,
deskewed strip, bounded full card, and deskewed bounded full card. It stops at
the first structurally accepted result, so successful fast-path cards do not
pay for MRZ localization, deskew, or full-image OCR.

TD1 and TD2 candidates are evaluated together rather than letting one format
block the other. Candidate scoring uses MRZ structure, line lengths, OCR
confidence, nationality position, and checksum quality; checksum is a quality
signal rather than a hard rejection because a single OCR error can invalidate
it while leaving the FIN readable. FIN values are accepted only from the
selected format's defined field. Card serial extraction remains optional.

OCR inputs are capped at 1600 pixels and Paddle retains the measured
960-pixel detection-side limit. See `benchmarks/README.md` for methodology,
private-fixture rules, method counts, and limitations.
