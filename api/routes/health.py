from typing import Annotated

from fastapi import APIRouter, Depends

from api.schemas import HealthResponse
from shared.config import Settings, get_settings


router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description=(
        "Returns API status, registered OCR services, and GPU worker capacity."
    ),
)
def health(
    settings: Annotated[Settings, Depends(get_settings)],
) -> HealthResponse:
    return HealthResponse(
        services=["id-fin"],
        ocr_max_concurrency=settings.ocr_max_concurrency,
        ocr_available_workers=None,
    )
