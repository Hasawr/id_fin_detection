import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from api.middleware.audit import AuditMiddleware
from api.routes import health, id_fin, passport
from api.schemas import ErrorDetail, IdFinResponse
from services.id_fin.detector import OCRProcessingError
from services.id_fin.service import get_id_fin_service
from shared.audit import get_audit_store


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.audit_store = get_audit_store()
    try:
        yield
    finally:
        if get_id_fin_service.cache_info().currsize:
            get_id_fin_service().close()
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
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response.model_dump(),
    )


app.include_router(health.router)
app.include_router(id_fin.router)
app.include_router(passport.router)
