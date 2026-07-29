import base64
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app
from services.id_fin.detector import OCRProcessingError
from services.id_fin.service import get_id_fin_service
from shared.audit import get_audit_store, reset_audit_store
from shared.config import Settings, get_settings


VALID_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
TEST_API_KEY = "test-api-key-that-is-at-least-32-characters"


def make_test_image(image_format: str) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), color="white").save(
        buffer,
        format=image_format,
    )
    return buffer.getvalue()


class FakeIDFinService:
    async def process(
        self,
        *,
        image_path: Path,
    ) -> dict[str, object]:
        return {
            "fin": "7ABC123" if image_path else None,
            "confidence": 0.99 if image_path else 0.0,
            "mrz_details": {
                "card_type": "new_card",
                "card_serial_number": "AA1234567",
                "line1": "SECRET-MRZ-LINE",
            }
            if image_path
            else None,
            "notes": [],
        }

    async def process_many(
        self,
        *,
        image_paths: list[Path],
    ) -> list[dict[str, object]]:
        return [
            await self.process(image_path=image_path)
            for image_path in image_paths
        ]

@pytest.fixture()
def client(tmp_path: Path):
    audit_db = tmp_path / "audit.db"
    payload_dir = tmp_path / "audit_payloads"
    reset_audit_store()
    store = get_audit_store(audit_db, payload_dir=payload_dir)

    def override_settings() -> Settings:
        return Settings(
            api_keys=(TEST_API_KEY,),
            use_gpu=False,
            save_ocr_debug_images=False,
            max_upload_bytes=1024 * 1024,
            max_image_pixels=1_000_000,
            max_batch_files=2,
            audit_db_path=audit_db,
            audit_payload_dir=payload_dir,
        )

    app.dependency_overrides[get_settings] = override_settings
    app.dependency_overrides[get_id_fin_service] = FakeIDFinService

    with TestClient(app) as test_client:
        app.state.audit_store = store
        yield test_client, store

    app.dependency_overrides.clear()
    reset_audit_store()


def test_health_does_not_require_authentication(client) -> None:
    test_client, _ = client
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "services": ["id-fin"],
    }


def test_health_does_not_initialize_ocr_service(client) -> None:
    test_client, _ = client

    def fail_if_initialized():
        raise AssertionError("Health must not initialize the OCR service.")

    app.dependency_overrides[get_id_fin_service] = fail_if_initialized
    response = test_client.get("/health")
    assert response.status_code == 200


def test_health_is_not_audited(client) -> None:
    test_client, store = client
    test_client.get("/health")
    assert store.summary(hours=None)["total"] == 0


def test_id_fin_requires_api_key(client) -> None:
    test_client, store = client
    response = test_client.post(
        "/v1/id-fin",
        files={"mrz": ("id.png", VALID_PNG, "image/png")},
    )
    assert response.status_code == 401
    summary = store.summary(hours=None)
    assert summary["total"] == 1
    assert summary["failed"] == 1


def test_id_fin_accepts_authorized_upload(client) -> None:
    test_client, store = client
    response = test_client.post(
        "/v1/id-fin",
        headers={"X-API-Key": TEST_API_KEY},
        files={"mrz": ("id.png", VALID_PNG, "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["service"] == "id-fin"
    assert response.json()["data"]["fin"] == "7ABC123"

    summary = store.summary(hours=None)
    assert summary["total"] == 1
    assert summary["succeeded"] == 1
    assert summary["ocr_detected"] == 1
    assert summary["ocr_not_found"] == 0
    assert summary["detection_rate"] == 100.0
    assert summary["by_service"][0]["name"] == "id-fin"

    events = store.recent_events(hours=None)
    assert len(events) == 1
    event = events[0]
    assert event.result_summary == (
        "FIN [REDACTED] · Serial [REDACTED] (0.99)"
    )
    assert event.response_body["data"]["fin"] == "[REDACTED]"
    assert (
        event.response_body["data"]["mrz_details"]["card_serial_number"]
        == "[REDACTED]"
    )
    # Upload metadata is recorded so the audit can say what was sent, but it
    # must carry no card data and no copy of the image itself.
    assert [part["field"] for part in event.request_files] == ["mrz"]
    assert all("saved_path" not in part for part in event.request_files)
    assert event.payload_dir is None
    assert "7ABC123" not in str(event.request_files)
    assert "7ABC123" not in str(event.response_body)
    assert "AA1234567" not in str(event.response_body)
    assert "SECRET-MRZ-LINE" not in str(event.response_body)


def test_audit_can_store_fin_and_serial_when_explicitly_enabled(client) -> None:
    test_client, store = client
    app.state.settings = replace(
        app.state.settings,
        audit_store_pii=True,
    )

    response = test_client.post(
        "/v1/id-fin",
        headers={"X-API-Key": TEST_API_KEY},
        files={"mrz": ("id.png", VALID_PNG, "image/png")},
    )

    assert response.status_code == 200
    event = store.recent_events(hours=None)[0]
    assert event.response_body["data"]["fin"] == "7ABC123"
    assert (
        event.response_body["data"]["mrz_details"]["card_serial_number"]
        == "AA1234567"
    )
    # Even in troubleshooting mode, full MRZ lines are unnecessary and must
    # never be copied into the audit database.
    assert "SECRET-MRZ-LINE" not in str(event.response_body)


def test_id_fin_batch_accepts_multiple_uploads(client) -> None:
    test_client, store = client
    response = test_client.post(
        "/v1/id-fin/batch",
        headers={"X-API-Key": TEST_API_KEY},
        files=[
            ("mrz", ("first.png", VALID_PNG, "image/png")),
            ("mrz", ("second.png", VALID_PNG, "image/png")),
        ],
    )
    assert response.status_code == 200
    assert response.json()["data"]["count"] == 2
    assert [
        result["file_name"] for result in response.json()["data"]["results"]
    ] == ["first.png", "second.png"]
    assert all(
        result["fin"] == "7ABC123"
        for result in response.json()["data"]["results"]
    )
    assert store.summary(hours=None)["total"] == 1
    assert store.summary(hours=None)["ocr_detected"] == 2


def test_audit_counts_http_ok_without_fin_as_not_found(client) -> None:
    _, store = client
    store.record(
        method="POST",
        path="/v1/id-fin",
        service="id-fin",
        status_code=200,
        latency_ms=10,
        response_body={
            "service": "id-fin",
            "data": {
                "fin": None,
                "confidence": 0.0,
                "mrz_details": None,
                "notes": ["MRZ not found"],
            },
            "error": None,
        },
    )

    summary = store.summary(hours=None)
    assert summary["succeeded"] == 1
    assert summary["failed"] == 0
    assert summary["ocr_detected"] == 0
    assert summary["ocr_not_found"] == 1
    assert summary["detection_rate"] == 0.0


def test_audit_counts_each_result_in_mixed_batch(client) -> None:
    _, store = client
    store.record(
        method="POST",
        path="/v1/id-fin/batch",
        service="id-fin",
        status_code=200,
        latency_ms=20,
        response_body={
            "service": "id-fin",
            "data": {
                "count": 2,
                "results": [
                    {"fin": "7ABC123"},
                    {"fin": None},
                ],
            },
            "error": None,
        },
    )

    summary = store.summary(hours=None)
    assert summary["ocr_detected"] == 1
    assert summary["ocr_not_found"] == 1
    assert summary["detection_rate"] == 50.0


def test_audit_clear_all_removes_events_and_saved_payloads(client) -> None:
    _, store = client
    payload_directory = store.payload_dir / "2026" / "07" / "request"
    payload_directory.mkdir(parents=True)
    (payload_directory / "id.png").write_bytes(VALID_PNG)
    relative_payload_directory = payload_directory.relative_to(
        store.payload_dir.parent
    )
    store.record(
        method="POST",
        path="/v1/id-fin",
        service="id-fin",
        status_code=200,
        latency_ms=10,
        response_body={
            "service": "id-fin",
            "data": {"fin": "7ABC123"},
            "error": None,
        },
        payload_dir=str(relative_payload_directory),
    )

    deleted = store.clear_all()

    assert deleted == {"events": 1, "payload_directories": 1}
    assert store.summary(hours=None)["total"] == 0
    assert not payload_directory.exists()
    assert store.payload_dir.is_dir()


def test_id_fin_batch_enforces_file_limit(client) -> None:
    test_client, _ = client
    response = test_client.post(
        "/v1/id-fin/batch",
        headers={"X-API-Key": TEST_API_KEY},
        files=[
            ("mrz", ("first.png", VALID_PNG, "image/png")),
            ("mrz", ("second.png", VALID_PNG, "image/png")),
            ("mrz", ("third.png", VALID_PNG, "image/png")),
        ],
    )
    assert response.status_code == 413


def test_id_fin_requires_an_image(client) -> None:
    test_client, _ = client
    response = test_client.post(
        "/v1/id-fin",
        headers={"X-API-Key": TEST_API_KEY},
    )
    assert response.status_code == 422


def test_id_fin_rejects_invalid_api_key(client) -> None:
    test_client, store = client
    response = test_client.post(
        "/v1/id-fin",
        headers={"X-API-Key": "wrong-key"},
        files={"mrz": ("id.png", VALID_PNG, "image/png")},
    )
    assert response.status_code == 401
    events = store.recent_events(hours=None)
    assert len(events) == 1
    assert events[0].success is False
    assert events[0].status_code == 401
    assert events[0].error_code == "Missing or invalid API key."
    assert events[0].result_summary == "Missing or invalid API key."
    assert events[0].response_body["detail"] == "Missing or invalid API key."


def test_id_fin_rejects_missing_file_records_validation_reason(client) -> None:
    test_client, store = client
    response = test_client.post(
        "/v1/id-fin",
        headers={"X-API-Key": TEST_API_KEY},
    )
    assert response.status_code == 422
    events = store.recent_events(hours=None)
    assert len(events) == 1
    assert events[0].success is False
    assert events[0].status_code == 422
    assert events[0].error_code is not None
    assert "Validation:" in events[0].error_code
    assert "Validation:" in events[0].result_summary
    assert events[0].response_body["detail"].startswith("Validation:")


def test_id_fin_rejects_different_length_api_key(client) -> None:
    test_client, _ = client
    response = test_client.post(
        "/v1/id-fin",
        headers={"X-API-Key": "x"},
        files={"mrz": ("id.png", VALID_PNG, "image/png")},
    )
    assert response.status_code == 401


def test_id_fin_rejects_unsupported_image_type(client) -> None:
    test_client, _ = client
    response = test_client.post(
        "/v1/id-fin",
        headers={"X-API-Key": TEST_API_KEY},
        files={"mrz": ("id.gif", b"GIF89a", "image/gif")},
    )
    assert response.status_code == 415


def test_id_fin_rejects_webp_even_with_image_content_type(client) -> None:
    test_client, _ = client
    response = test_client.post(
        "/v1/id-fin",
        headers={"X-API-Key": TEST_API_KEY},
        files={"mrz": ("id.webp", b"RIFFxxxxWEBP", "image/webp")},
    )
    assert response.status_code == 415


@pytest.mark.parametrize(
    ("file_name", "content_type", "image_format"),
    [
        ("id.png", "image/png", "PNG"),
        ("id.jpg", "image/jpeg", "JPEG"),
        ("id.bmp", "image/bmp", "BMP"),
    ],
)
def test_id_fin_accepts_supported_image_formats(
    client,
    file_name: str,
    content_type: str,
    image_format: str,
) -> None:
    test_client, _ = client
    response = test_client.post(
        "/v1/id-fin",
        headers={"X-API-Key": TEST_API_KEY},
        files={
            "mrz": (
                file_name,
                make_test_image(image_format),
                content_type,
            )
        },
    )
    assert response.status_code == 200


def test_id_fin_rejects_empty_image(client) -> None:
    test_client, _ = client
    response = test_client.post(
        "/v1/id-fin",
        headers={"X-API-Key": TEST_API_KEY},
        files={"mrz": ("id.png", b"", "image/png")},
    )
    assert response.status_code == 400


def test_id_fin_reports_ocr_engine_failure(client) -> None:
    test_client, _ = client

    class FailingIDFinService:
        async def process(self, *, image_path: Path) -> dict[str, object]:
            del image_path
            raise OCRProcessingError("engine failed")

    app.dependency_overrides[get_id_fin_service] = FailingIDFinService
    response = test_client.post(
        "/v1/id-fin",
        headers={"X-API-Key": TEST_API_KEY},
        files={"mrz": ("id.png", VALID_PNG, "image/png")},
    )

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "ocr_processing_failed"


def test_id_fin_batch_reports_ocr_engine_failure(client) -> None:
    test_client, _ = client

    class FailingIDFinService:
        async def process_many(
            self,
            *,
            image_paths: list[Path],
        ) -> list[dict[str, object]]:
            del image_paths
            raise OCRProcessingError("engine failed")

    app.dependency_overrides[get_id_fin_service] = FailingIDFinService
    response = test_client.post(
        "/v1/id-fin/batch",
        headers={"X-API-Key": TEST_API_KEY},
        files=[("mrz", ("id.png", VALID_PNG, "image/png"))],
    )

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "ocr_processing_failed"


def test_id_fin_rejects_oversized_upload(client) -> None:
    test_client, _ = client
    limited_settings = replace(app.state.settings, max_upload_bytes=1)
    app.dependency_overrides[get_settings] = lambda: limited_settings

    response = test_client.post(
        "/v1/id-fin",
        headers={"X-API-Key": TEST_API_KEY},
        files={"mrz": ("id.png", VALID_PNG, "image/png")},
    )

    assert response.status_code == 413


def test_id_fin_rejects_oversized_image_dimensions(
    client,
) -> None:
    test_client, _ = client
    limited_settings = replace(app.state.settings, max_image_pixels=1)
    app.dependency_overrides[get_settings] = lambda: limited_settings

    response = test_client.post(
        "/v1/id-fin",
        headers={"X-API-Key": TEST_API_KEY},
        files={
            "mrz": (
                "id.png",
                make_test_image("PNG"),
                "image/png",
            )
        },
    )

    assert response.status_code == 413


def test_id_fin_rejects_invalid_image_content(client) -> None:
    test_client, _ = client
    response = test_client.post(
        "/v1/id-fin",
        headers={"X-API-Key": TEST_API_KEY},
        files={"mrz": ("id.png", b"not-an-image", "image/png")},
    )
    assert response.status_code == 422


def test_id_fin_reports_unconfigured_authentication(client) -> None:
    test_client, _ = client
    unconfigured_settings = replace(app.state.settings, api_keys=())
    app.dependency_overrides[get_settings] = lambda: unconfigured_settings

    response = test_client.post(
        "/v1/id-fin",
        files={"mrz": ("id.png", VALID_PNG, "image/png")},
    )

    assert response.status_code == 503


def test_passport_is_an_authenticated_stub(client) -> None:
    test_client, store = client
    response = test_client.post(
        "/v1/passport",
        headers={"X-API-Key": TEST_API_KEY},
    )
    assert response.status_code == 501
    assert response.json()["error"]["code"] == "not_implemented"

    events = store.recent_events(hours=None)
    assert len(events) == 1
    assert events[0].service == "passport"
    assert events[0].success is False
    assert events[0].error_code == "not_implemented"


def test_batch_enforces_aggregate_byte_limit(client) -> None:
    test_client, _ = client
    limited_settings = replace(
        app.state.settings,
        max_batch_bytes=len(VALID_PNG),
    )
    app.state.settings = limited_settings
    app.dependency_overrides[get_settings] = lambda: limited_settings
    response = test_client.post(
        "/v1/id-fin/batch",
        headers={"X-API-Key": TEST_API_KEY},
        files=[
            ("mrz", ("first.png", VALID_PNG, "image/png")),
            ("mrz", ("second.png", VALID_PNG, "image/png")),
        ],
    )
    assert response.status_code == 413


def test_raw_payload_retention_requires_opt_in(client) -> None:
    test_client, store = client
    app.state.settings = replace(
        app.state.settings,
        audit_store_payloads=True,
    )
    response = test_client.post(
        "/v1/id-fin",
        headers={"X-API-Key": TEST_API_KEY},
        files={"mrz": ("id.png", VALID_PNG, "image/png")},
    )
    assert response.status_code == 200

    event = store.recent_events(hours=None)[0]
    assert len(event.request_files) == 1
    saved = store.payload_dir.parent / event.request_files[0]["saved_path"]
    assert saved.read_bytes() == VALID_PNG


def test_rejected_payload_is_never_retained_when_opted_in(client) -> None:
    test_client, store = client
    app.state.settings = replace(
        app.state.settings,
        audit_store_payloads=True,
    )
    response = test_client.post(
        "/v1/id-fin",
        headers={"X-API-Key": "invalid"},
        files={"mrz": ("id.png", VALID_PNG, "image/png")},
    )
    assert response.status_code == 401
    event = store.recent_events(hours=None)[0]
    assert event.request_files == []
    assert event.payload_dir is None


def test_api_docs_are_always_visible(client) -> None:
    test_client, _ = client
    assert test_client.get("/docs").status_code == 200
    assert test_client.get("/openapi.json").status_code == 200


def test_request_part_metadata_is_recorded_without_storing_payloads(
    client,
) -> None:
    """The audit answers "what did they send?" without keeping the image.

    File names, media types and sizes are not the sensitive part; the image
    bytes are. Metadata is therefore recorded even with AUDIT_STORE_PAYLOADS
    off, which is the production default.
    """
    test_client, store = client
    assert app.state.settings.audit_store_payloads is False

    response = test_client.post(
        "/v1/id-fin",
        headers={"X-API-Key": TEST_API_KEY},
        files={"mrz": ("front-side.png", VALID_PNG, "image/png")},
    )
    assert response.status_code == 200

    event = store.recent_events(hours=None)[0]
    assert len(event.request_files) == 1
    part = event.request_files[0]
    assert part["field"] == "mrz"
    assert part["file_name"] == "front-side.png"
    assert part["content_type"] == "image/png"
    assert part["size_bytes"] == len(VALID_PNG)
    # The bytes themselves must not have been kept.
    assert "saved_path" not in part
    assert event.payload_dir is None


def test_rejected_request_records_no_part_metadata(client) -> None:
    """An unauthenticated caller must not write into the audit log at all.

    Without this, anyone who can reach the endpoint could persist arbitrary
    attacker-chosen file names by sending requests that are then rejected.
    """
    test_client, store = client

    response = test_client.post(
        "/v1/id-fin",
        headers={"X-API-Key": "invalid"},
        files={"mrz": ("attacker-chosen-name.png", VALID_PNG, "image/png")},
    )
    assert response.status_code == 401

    event = store.recent_events(hours=None)[0]
    assert event.request_files == []
    assert event.payload_dir is None
