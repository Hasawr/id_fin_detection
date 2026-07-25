import logging
import time
from collections.abc import Callable

import httpx
from pydantic import ValidationError

from api.schemas import IdFinData


logger = logging.getLogger(__name__)
UploadJob = tuple[str, bytes, str]
TimedResult = tuple[str, IdFinData, float]
ProgressHandler = Callable[[int, int], None]


class OCRApiError(RuntimeError):
    """A safe, user-facing failure from the local OCR API."""


def create_ocr_api_client(
    base_url: str,
    api_key: str,
    *,
    transport: httpx.BaseTransport | None = None,
) -> httpx.Client:
    timeout = httpx.Timeout(
        180.0,
        connect=5.0,
        read=180.0,
        write=30.0,
        pool=5.0,
    )
    return httpx.Client(
        base_url=base_url.rstrip("/"),
        headers={"X-API-Key": api_key},
        timeout=timeout,
        transport=transport,
    )


def is_ocr_api_healthy(base_url: str) -> bool:
    try:
        response = httpx.get(
            f"{base_url.rstrip('/')}/health",
            timeout=httpx.Timeout(3.0, connect=1.0),
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        return False
    return isinstance(payload, dict) and payload.get("status") == "ok"


def request_id_fin(
    client: httpx.Client,
    *,
    file_name: str,
    image_bytes: bytes,
    content_type: str,
) -> IdFinData:
    try:
        response = client.post(
            "/v1/id-fin",
            files={
                "mrz": (
                    file_name,
                    image_bytes,
                    content_type,
                )
            },
        )
    except httpx.TimeoutException as exc:
        raise OCRApiError(
            "The OCR API timed out while processing this image."
        ) from exc
    except httpx.RequestError as exc:
        raise OCRApiError(
            "The OCR API is unavailable. Check that the backend is running."
        ) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise OCRApiError("The OCR API returned an invalid response.") from exc

    if response.is_error:
        error = payload.get("error") if isinstance(payload, dict) else None
        detail = payload.get("detail") if isinstance(payload, dict) else None
        message = (
            error.get("message")
            if isinstance(error, dict)
            else detail
        )
        raise OCRApiError(
            str(message)
            if message
            else f"The OCR API returned HTTP {response.status_code}."
        )

    if not isinstance(payload, dict) or payload.get("service") != "id-fin":
        raise OCRApiError("The OCR API returned an unexpected response.")
    try:
        return IdFinData.model_validate(payload.get("data"))
    except ValidationError as exc:
        raise OCRApiError("The OCR API returned invalid result data.") from exc


def process_id_fin_jobs(
    client: httpx.Client,
    jobs: list[UploadJob],
    *,
    handle_progress: ProgressHandler | None = None,
) -> list[TimedResult]:
    results: list[TimedResult] = []
    for file_name, image_bytes, content_type in jobs:
        started = time.perf_counter()
        try:
            result = request_id_fin(
                client,
                file_name=file_name,
                image_bytes=image_bytes,
                content_type=content_type,
            )
        except OCRApiError as exc:
            result = IdFinData(
                fin=None,
                confidence=0.0,
                notes=[f"Processing error: {exc}"],
            )
        except Exception:
            logger.exception("Unexpected Streamlit OCR client failure")
            result = IdFinData(
                fin=None,
                confidence=0.0,
                notes=["Processing error: Unexpected client failure."],
            )
        elapsed = time.perf_counter() - started
        results.append((file_name, result, elapsed))
        if handle_progress is not None:
            handle_progress(len(results), len(jobs))
    return results
