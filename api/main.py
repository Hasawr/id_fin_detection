import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.middleware.audit import AuditMiddleware
from api.routes import health, id_fin, passport
from shared.audit import get_audit_store


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.audit_store = get_audit_store()
    yield


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
app.include_router(health.router)
app.include_router(id_fin.router)
app.include_router(passport.router)
