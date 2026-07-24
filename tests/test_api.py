import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app
from services.id_fin.service import get_id_fin_service
from shared.audit import get_audit_store, reset_audit_store
from shared.config import Settings, get_settings


VALID_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


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
            api_keys=("test-key",),
            use_gpu=False,
            debug=False,
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
        "services": ["id-fin", "passport"],
    }


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
        headers={"X-API-Key": "test-key"},
        files={"mrz": ("id.png", VALID_PNG, "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["service"] == "id-fin"
    assert response.json()["data"]["fin"] == "7ABC123"

    summary = store.summary(hours=None)
    assert summary["total"] == 1
    assert summary["succeeded"] == 1
    assert summary["by_service"][0]["name"] == "id-fin"

    events = store.recent_events(hours=None)
    assert len(events) == 1
    event = events[0]
    assert event.result_summary == "FIN 7ABC123 · Serial AA1234567 (0.99)"
    assert event.response_body["data"]["fin"] == "7ABC123"
    assert (
        event.response_body["data"]["mrz_details"]["card_serial_number"]
        == "AA1234567"
    )
    assert len(event.request_files) == 1
    assert event.request_files[0]["file_name"] == "id.png"
    saved = store.payload_dir.parent / event.request_files[0]["saved_path"]
    assert saved.exists()
    assert saved.read_bytes() == VALID_PNG


def test_id_fin_batch_accepts_multiple_uploads(client) -> None:
    test_client, store = client
    response = test_client.post(
        "/v1/id-fin/batch",
        headers={"X-API-Key": "test-key"},
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


def test_id_fin_batch_enforces_file_limit(client) -> None:
    test_client, _ = client
    response = test_client.post(
        "/v1/id-fin/batch",
        headers={"X-API-Key": "test-key"},
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
        headers={"X-API-Key": "test-key"},
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


def test_id_fin_rejects_invalid_image_content(client) -> None:
    test_client, _ = client
    response = test_client.post(
        "/v1/id-fin",
        headers={"X-API-Key": "test-key"},
        files={"mrz": ("id.png", b"not-an-image", "image/png")},
    )
    assert response.status_code == 422


def test_passport_is_an_authenticated_stub(client) -> None:
    test_client, store = client
    response = test_client.post(
        "/v1/passport",
        headers={"X-API-Key": "test-key"},
    )
    assert response.status_code == 501
    assert response.json()["error"]["code"] == "not_implemented"

    events = store.recent_events(hours=None)
    assert len(events) == 1
    assert events[0].service == "passport"
    assert events[0].success is False
    assert events[0].error_code == "not_implemented"
