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
    description="Versioned OCR endpoints for Azerbaijani identity documents.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(AuditMiddleware)
app.include_router(health.router)
app.include_router(id_fin.router)
app.include_router(passport.router)
