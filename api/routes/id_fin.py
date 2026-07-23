from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from api.auth import require_api_key
from api.schemas import ServiceResponse
from services.id_fin.service import IDFinService, get_id_fin_service
from shared.config import Settings, get_settings
from shared.image_io import save_upload


router = APIRouter(prefix="/v1", tags=["id-fin"])


@router.post("/id-fin", response_model=ServiceResponse)
async def detect_id_fin(
    api_key: Annotated[str, Depends(require_api_key)],
    service: Annotated[IDFinService, Depends(get_id_fin_service)],
    mrz: Annotated[UploadFile, File(description="ID card MRZ-side image")],
) -> ServiceResponse:
    del api_key

    with TemporaryDirectory(prefix="ocr-id-fin-") as directory:
        temporary_directory = Path(directory)
        image_path = await save_upload(mrz, temporary_directory, "mrz")
        data = await service.process(image_path=image_path)

    return ServiceResponse(service="id-fin", data=data)


@router.post("/id-fin/batch", response_model=ServiceResponse)
async def detect_id_fin_batch(
    api_key: Annotated[str, Depends(require_api_key)],
    service: Annotated[IDFinService, Depends(get_id_fin_service)],
    settings: Annotated[Settings, Depends(get_settings)],
    mrz: Annotated[
        list[UploadFile],
        File(description="One or more ID card MRZ-side images"),
    ],
) -> ServiceResponse:
    del api_key
    if len(mrz) > settings.max_batch_files:
        for upload in mrz:
            await upload.close()
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"A batch may contain at most {settings.max_batch_files} images.",
        )

    with TemporaryDirectory(prefix="ocr-id-fin-batch-") as directory:
        temporary_directory = Path(directory)
        image_paths = [
            await save_upload(upload, temporary_directory, f"mrz_{index}")
            for index, upload in enumerate(mrz)
        ]
        detected_results = await service.process_many(image_paths=image_paths)

    results = [
        {
            "index": index,
            "file_name": Path(upload.filename or f"mrz_{index}").name,
            **detected_result,
        }
        for index, (upload, detected_result) in enumerate(
            zip(mrz, detected_results, strict=True)
        )
    ]
    return ServiceResponse(
        service="id-fin",
        data={"count": len(results), "results": results},
    )
