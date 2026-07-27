import os
import sys
import base64
import logging
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).parent))

from fin_detector.batch_processor import BatchFINProcessor
from fin_detector.detector import FINDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("server")

# Global processor instance
batch_processor: Optional[BatchFINProcessor] = None
DEFAULT_WORKERS = int(os.environ.get("BATCH_WORKERS", "4"))
DEFAULT_USE_GPU = os.environ.get("USE_GPU", "true").lower() in ("true", "1", "yes")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager to start and shutdown worker process pool gracefully."""
    global batch_processor
    logger.info(f"Initializing BatchFINProcessor pool with {DEFAULT_WORKERS} workers (GPU={DEFAULT_USE_GPU})...")
    batch_processor = BatchFINProcessor(max_workers=DEFAULT_WORKERS, use_gpu=DEFAULT_USE_GPU)
    yield
    if batch_processor:
        logger.info("Shutting down BatchFINProcessor pool...")
        batch_processor.shutdown()

app = FastAPI(
    title="Azerbaijani ID Card FIN & ID Detection API",
    description="High-performance, parallel production REST API to extract FIN & ID numbers from MRZ & VIZ document images.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files & Debug Images
debug_dir = Path(__file__).parent / "debug_output"
debug_dir.mkdir(exist_ok=True)
app.mount("/debug_output", StaticFiles(directory=str(debug_dir)), name="debug_output")

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Pydantic Schemas for Base64 Request Body
class Base64ImageItem(BaseModel):
    id: str = Field(..., description="Unique image identifier or filename")
    base64_data: str = Field(..., description="Base64 encoded image string (with or without data URL header)")

class Base64BatchRequest(BaseModel):
    images: List[Base64ImageItem]
    mode: str = Field("mrz", description="Extraction mode: 'mrz', 'viz', or 'both'")


@app.get("/", include_in_schema=False)
async def serve_index():
    """Serve single-page Web Dashboard at root URL."""
    return FileResponse(static_dir / "index.html")

@app.get("/health", summary="Health check endpoint")
async def health_check():
    """Verify service health and worker pool status."""
    return {
        "status": "healthy",
        "service": "id_fin_detection",
        "gpu_enabled": DEFAULT_USE_GPU,
        "worker_pool_active": batch_processor is not None and batch_processor.executor is not None
    }

@app.get("/api/v1/info", summary="System state and configuration")
async def system_info():
    """Get active worker pool configuration and system specs."""
    return {
        "max_workers": DEFAULT_WORKERS,
        "use_gpu": DEFAULT_USE_GPU,
        "python_version": sys.version,
        "pid": os.getpid()
    }

@app.post("/api/v1/detect_batch", summary="Batch process up to 100 images in parallel")
async def detect_batch(
    files: List[UploadFile] = File(None, description="List of image files to process"),
    mode: str = Form("mrz", description="Extraction mode: 'mrz', 'viz'"),
):
    """
    Process up to 100 image files concurrently using worker process pool.
    Returns structured results per file, total timing, and throughput (images/sec).
    """
    if not batch_processor:
        raise HTTPException(status_code=500, detail="Worker process pool is uninitialized.")
        
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="No files uploaded in request.")
        
    if len(files) > 150:
        raise HTTPException(status_code=400, detail="Maximum batch limit is 150 images per request.")

    # Read uploaded file bytes
    items = []
    for file in files:
        content = await file.read()
        items.append({
            "id": file.filename or f"img_{len(items)}",
            "input": content
        })

    # Execute batch in parallel worker pool
    result = batch_processor.process_batch(items, mode=mode)
    return result


@app.post("/api/v1/detect_batch_json", summary="Batch process base64 images in JSON payload")
async def detect_batch_json(request: Base64BatchRequest):
    """
    Process base64 encoded images sent via JSON array.
    """
    if not batch_processor:
        raise HTTPException(status_code=500, detail="Worker process pool is uninitialized.")
        
    if not request.images:
        raise HTTPException(status_code=400, detail="No images provided in JSON request.")
        
    if len(request.images) > 150:
        raise HTTPException(status_code=400, detail="Maximum batch limit is 150 images per request.")

    items = []
    for item in request.images:
        b64_str = item.base64_data
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        try:
            raw_bytes = base64.b64decode(b64_str)
            items.append({
                "id": item.id,
                "input": raw_bytes
            })
        except Exception as e:
            items.append({
                "id": item.id,
                "input": b""  # Empty bytes will trigger individual failure handling gracefully
            })

    result = batch_processor.process_batch(items, mode=request.mode)
    return result


@app.post("/api/v1/detect_single", summary="Process a single image")
async def detect_single(
    file: UploadFile = File(..., description="Single ID card image"),
    mode: str = Form("mrz", description="Extraction mode: 'mrz', 'viz'")
):
    """Fast-path endpoint for single image extraction."""
    if not batch_processor:
        raise HTTPException(status_code=500, detail="Worker process pool is uninitialized.")
        
    content = await file.read()
    item = [{"id": file.filename or "single_image", "input": content}]
    
    batch_res = batch_processor.process_batch(item, mode=mode)
    if batch_res["results"]:
        return batch_res["results"][0]
    raise HTTPException(status_code=500, detail="Processing failed.")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
