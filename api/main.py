import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status as http_status
from fastapi.responses import JSONResponse

from api.middleware.audit import AuditMiddleware
from api.routes import health, id_fin, passport, status as status_page
from api.schemas import ErrorDetail, IdFinResponse
from services.id_fin.detector import OCRProcessingError
from services.id_fin.service import get_id_fin_service
from shared.audit import get_audit_store
from shared.config import get_settings, validate_security_settings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings_provider = app.dependency_overrides.get(
        get_settings,
        get_settings,
    )
    settings = settings_provider()
    validate_security_settings(settings)
    app.state.settings = settings
    app.state.audit_store = get_audit_store(
        settings.audit_db_path,
        payload_dir=settings.audit_payload_dir,
    )
    app.state.audit_store.prune(
        retention_days=settings.audit_retention_days,
        max_payload_bytes=settings.audit_max_payload_bytes,
    )
    try:
        yield
    finally:
        if get_id_fin_service.cache_info().currsize:
            service = get_id_fin_service()
            await asyncio.to_thread(service.close)
            get_id_fin_service.cache_clear()


app = FastAPI(
    title="Multi-service OCR API",
    description=(
        "Versioned OCR endpoints for Azerbaijani identity documents.\n\n"
        "**Authentication:** send a configured key in the `X-API-Key` header.\n\n"
        "**ID FIN service:** extracts the personal FIN from the MRZ side of an "
        "ID card. Validated document serials are returned as "
        "`mrz_details.card_serial_number` (`AA`/`AB` + 7 digits for new TD1 "
        "cards; numeric document number for older TD2 cards)."
    ),
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {
            "name": "id-fin",
            "description": "Azerbaijani ID card FIN and card-serial OCR.",
        },
        {
            "name": "passport",
            "description": "Passport OCR placeholder (not implemented yet).",
        },
        {
            "name": "health",
            "description": "Service health checks.",
        },
        {
            "name": "status",
            "description": (
                "Public sanitized status page (disabled unless "
                "PUBLIC_STATUS_ENABLED=true)."
            ),
        },
    ],
)
app.add_middleware(AuditMiddleware)


@app.exception_handler(OCRProcessingError)
async def handle_ocr_processing_error(
    request: Request,
    exc: OCRProcessingError,
) -> JSONResponse:
    del request, exc
    response = IdFinResponse(
        service="id-fin",
        error=ErrorDetail(
            code="ocr_processing_failed",
            message="The OCR engine could not process the image.",
        ),
    )
    return JSONResponse(
        status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response.model_dump(),
    )


app.include_router(health.router)
app.include_router(status_page.router)
app.include_router(id_fin.router)
app.include_router(passport.router)
