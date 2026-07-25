from fastapi import APIRouter, Depends, status

from api.auth import require_api_key
from api.schemas import ErrorDetail, ServiceResponse


router = APIRouter(
    prefix="/v1",
    tags=["passport"],
    dependencies=[Depends(require_api_key)],
)


@router.post(
    "/passport",
    response_model=ServiceResponse,
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    responses={501: {"model": ServiceResponse}},
    summary="Passport OCR (not implemented)",
    description="Placeholder endpoint. Returns HTTP 501 until passport OCR is available.",
)
async def detect_passport(
) -> ServiceResponse:
    return ServiceResponse(
        service="passport",
        error=ErrorDetail(
            code="not_implemented",
            message="Passport OCR is not implemented yet.",
        ),
    )
