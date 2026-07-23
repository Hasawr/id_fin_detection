from fastapi import APIRouter

from api.schemas import HealthResponse


router = APIRouter(tags=["platform"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(services=["id-fin", "passport"])
