from pathlib import Path

from fastapi import HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

from shared.config import Settings


ALLOWED_CONTENT_TYPES = {
    "image/bmp",
    "image/jpeg",
    "image/png",
}
ALLOWED_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png"}
CHUNK_SIZE = 1024 * 1024


async def save_upload(
    upload: UploadFile,
    directory: Path,
    stem: str,
    settings: Settings,
) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    if upload.content_type not in ALLOWED_CONTENT_TYPES or suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only BMP, JPEG, and PNG images are supported.",
        )

    destination = directory / f"{stem}{suffix}"
    bytes_written = 0

    try:
        with destination.open("wb") as output:
            while chunk := await upload.read(CHUNK_SIZE):
                bytes_written += len(chunk)
                if bytes_written > settings.max_upload_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail="Uploaded image exceeds the configured size limit.",
                    )
                output.write(chunk)
    finally:
        await upload.close()

    if bytes_written == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image is empty.",
        )

    try:
        with Image.open(destination) as image:
            width, height = image.size
            if width * height > settings.max_image_pixels:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail="Uploaded image dimensions exceed the configured limit.",
                )
            image.verify()
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Uploaded file is not a valid image.",
        ) from None

    return destination
