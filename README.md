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

Start the API from the repository root:

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
    "mrz_details": {},
    "notes": []
  },
  "error": null
}
```

For multiple cards, repeat the `mrz` multipart field:

```powershell
curl.exe -X POST "http://localhost:8000/v1/id-fin/batch" `
  -H "X-API-Key: first-client-key" `
  -F "mrz=@C:\images\first-back.jpg" `
  -F "mrz=@C:\images\second-back.jpg"
```

Batch responses contain `data.count` and an ordered `data.results` list. Each
result includes `index`, `file_name`, `fin`, `confidence`, `mrz_details`, and
`notes`. OCR runs sequentially on the shared GPU to avoid concurrent model
access and GPU-memory spikes.

Each image is limited by `MAX_UPLOAD_BYTES` (10 MiB by default) and
`MAX_IMAGE_PIXELS` (25 megapixels by default). File signatures are verified
before OCR. A batch accepts at most `MAX_BATCH_FILES` images (10 by default).
Uploaded files are stored in a request-specific temporary directory and
removed after processing.

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
```

The HTTP route handles transport and validation. Each package under
`services/` owns its OCR implementation. `IDFinService` serializes the OCR
result and runs the blocking detector in a worker thread with a lock around
the shared PaddleOCR instance.

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
