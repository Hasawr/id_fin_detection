from typing import Annotated

from fastapi import APIRouter, Depends

from api.schemas import HealthResponse
from services.id_fin.service import IDFinService, get_id_fin_service


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
    service: Annotated[IDFinService, Depends(get_id_fin_service)],
) -> HealthResponse:
    concurrency = service.concurrency_info()
    return HealthResponse(
        services=["id-fin", "passport"],
        ocr_max_concurrency=int(concurrency["max_concurrency"]),
        ocr_available_workers=int(concurrency["available_workers"]),
    )
