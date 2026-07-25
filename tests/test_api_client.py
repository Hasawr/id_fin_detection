import httpx
import pytest

from demos.api_client import (
    OCRApiError,
    create_ocr_api_client,
    process_id_fin_jobs,
    request_id_fin,
)


def _success_response() -> dict[str, object]:
    return {
        "service": "id-fin",
        "version": "v1",
        "data": {
            "fin": "1ABC234",
            "confidence": 0.96,
            "mrz_details": None,
            "notes": [],
        },
        "error": None,
    }


def test_api_client_uploads_image_and_parses_success() -> None:
    def handle_request(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-API-Key"] == "test-key"
        assert request.url.path == "/v1/id-fin"
        assert b'filename="card.jpg"' in request.read()
        return httpx.Response(200, json=_success_response())

    with create_ocr_api_client(
        "http://test",
        "test-key",
        transport=httpx.MockTransport(handle_request),
    ) as client:
        result = request_id_fin(
            client,
            file_name="card.jpg",
            image_bytes=b"image",
            content_type="image/jpeg",
        )

    assert result.fin == "1ABC234"
    assert result.confidence == 0.96


def test_api_client_reports_timeout() -> None:
    def handle_request(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with create_ocr_api_client(
        "http://test",
        "test-key",
        transport=httpx.MockTransport(handle_request),
    ) as client:
        with pytest.raises(OCRApiError, match="timed out"):
            request_id_fin(
                client,
                file_name="card.jpg",
                image_bytes=b"image",
                content_type="image/jpeg",
            )


def test_api_client_reports_connection_failure() -> None:
    def handle_request(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    with create_ocr_api_client(
        "http://test",
        "test-key",
        transport=httpx.MockTransport(handle_request),
    ) as client:
        with pytest.raises(OCRApiError, match="unavailable"):
            request_id_fin(
                client,
                file_name="card.jpg",
                image_bytes=b"image",
                content_type="image/jpeg",
            )


def test_api_client_rejects_malformed_response() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"service": "id-fin"})
    )
    with create_ocr_api_client(
        "http://test",
        "test-key",
        transport=transport,
    ) as client:
        with pytest.raises(OCRApiError, match="invalid result data"):
            request_id_fin(
                client,
                file_name="card.jpg",
                image_bytes=b"image",
                content_type="image/jpeg",
            )


def test_api_client_continues_after_per_image_failure() -> None:
    request_count = 0

    def handle_request(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            return httpx.Response(
                500,
                json={
                    "service": "id-fin",
                    "data": None,
                    "error": {
                        "code": "ocr_processing_failed",
                        "message": "OCR failed.",
                    },
                },
            )
        return httpx.Response(200, json=_success_response())

    progress: list[tuple[int, int]] = []
    with create_ocr_api_client(
        "http://test",
        "test-key",
        transport=httpx.MockTransport(handle_request),
    ) as client:
        results = process_id_fin_jobs(
            client,
            [
                ("bad.jpg", b"bad", "image/jpeg"),
                ("good.jpg", b"good", "image/jpeg"),
            ],
            handle_progress=lambda completed, total: progress.append(
                (completed, total)
            ),
        )

    assert results[0][1].fin is None
    assert results[0][1].notes == ["Processing error: OCR failed."]
    assert results[1][1].fin == "1ABC234"
    assert progress == [(1, 2), (2, 2)]
