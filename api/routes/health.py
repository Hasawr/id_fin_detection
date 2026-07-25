from fastapi import APIRouter

from api.schemas import HealthResponse


router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description=(
        "Returns API status and registered OCR services."
    ),
)
def health() -> HealthResponse:
    return HealthResponse(
        services=["id-fin"],
    )
