from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from api.auth import require_api_key
from api.schemas import (
    IdFinBatchData,
    IdFinBatchItem,
    IdFinBatchResponse,
    IdFinResponse,
)
from services.id_fin.service import IDFinService, get_id_fin_service
from shared.config import Settings, get_settings
from shared.image_io import save_upload


router = APIRouter(
    prefix="/v1",
    tags=["id-fin"],
    dependencies=[Depends(require_api_key)],
)


@router.post(
    "/id-fin",
    response_model=IdFinResponse,
    summary="Detect FIN and card serial from one MRZ image",
    description=(
        "Upload the MRZ side of an Azerbaijani ID card. "
        "Returns the personal FIN when detected. "
        "`mrz_details.card_serial_number` is returned for new TD1 cards "
        "(`AA`/`AB` + 7 digits) and older TD2 cards (numeric document number) "
        "when the value can be validated."
    ),
    responses={
        400: {"description": "Empty upload"},
        401: {"description": "Missing or invalid API key"},
        413: {"description": "Upload too large"},
        415: {"description": "Unsupported image type"},
        422: {"description": "Invalid image content"},
        500: {"description": "OCR engine failed to process the image"},
    },
)
async def detect_id_fin(
    service: Annotated[IDFinService, Depends(get_id_fin_service)],
    settings: Annotated[Settings, Depends(get_settings)],
    mrz: Annotated[UploadFile, File(description="ID card MRZ-side image")],
) -> IdFinResponse:
    with TemporaryDirectory(prefix="ocr-id-fin-") as directory:
        temporary_directory = Path(directory)
        image_path = await save_upload(
            mrz,
            temporary_directory,
            "mrz",
            settings,
        )
        data = await service.process(image_path=image_path)

    return IdFinResponse(service="id-fin", data=data)


@router.post(
    "/id-fin/batch",
    response_model=IdFinBatchResponse,
    summary="Detect FIN and card serial from multiple MRZ images",
    description=(
        "Upload one or more MRZ-side images in a single request. "
        "Each result includes `fin`, `confidence`, and `mrz_details` "
        "(with `card_serial_number` for validated new-card and older-card "
        "serials). "
        "Images in a batch are processed sequentially by one OCR engine."
    ),
    responses={
        400: {"description": "An uploaded image is empty"},
        401: {"description": "Missing or invalid API key"},
        413: {"description": "Too many files or upload too large"},
        415: {"description": "Unsupported image type"},
        422: {"description": "Invalid image content"},
        500: {"description": "OCR engine failed to process an image"},
    },
)
async def detect_id_fin_batch(
    service: Annotated[IDFinService, Depends(get_id_fin_service)],
    settings: Annotated[Settings, Depends(get_settings)],
    mrz: Annotated[
        list[UploadFile],
        File(description="One or more ID card MRZ-side images"),
    ],
) -> IdFinBatchResponse:
    if len(mrz) > settings.max_batch_files:
        for upload in mrz:
            await upload.close()
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"A batch may contain at most {settings.max_batch_files} images.",
        )

    with TemporaryDirectory(prefix="ocr-id-fin-batch-") as directory:
        temporary_directory = Path(directory)
        image_paths: list[Path] = []
        batch_bytes = 0
        for index, upload in enumerate(mrz):
            image_path = await save_upload(
                upload,
                temporary_directory,
                f"mrz_{index}",
                settings,
            )
            batch_bytes += image_path.stat().st_size
            if batch_bytes > settings.max_batch_bytes:
                for remaining_upload in mrz[index + 1 :]:
                    await remaining_upload.close()
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail="Batch exceeds the configured total size limit.",
                )
            image_paths.append(image_path)
        detected_results = await service.process_many(image_paths=image_paths)

    results = [
        IdFinBatchItem(
            index=index,
            file_name=Path(upload.filename or f"mrz_{index}").name,
            **detected_result,
        )
        for index, (upload, detected_result) in enumerate(
            zip(mrz, detected_results, strict=True)
        )
    ]
    return IdFinBatchResponse(
        service="id-fin",
        data=IdFinBatchData(count=len(results), results=results),
    )
