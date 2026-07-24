from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from api.auth import require_api_key
from api.schemas import ErrorDetail, ServiceResponse


router = APIRouter(prefix="/v1", tags=["passport"])


@router.post(
    "/passport",
    response_model=ServiceResponse,
    responses={501: {"model": ServiceResponse}},
    summary="Passport OCR (not implemented)",
    description="Placeholder endpoint. Returns HTTP 501 until passport OCR is available.",
)
async def detect_passport(
    api_key: Annotated[str, Depends(require_api_key)],
) -> JSONResponse:
    del api_key
    response = ServiceResponse(
        service="passport",
        error=ErrorDetail(
            code="not_implemented",
            message="Passport OCR is not implemented yet.",
        ),
    )
    return JSONResponse(status_code=501, content=jsonable_encoder(response))
