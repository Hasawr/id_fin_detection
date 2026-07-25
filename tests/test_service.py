import asyncio
from pathlib import Path
from threading import Event

import pytest

from services.id_fin import FINDetectionOutput, MRZResult
from services.id_fin.detector import OCRProcessingError
from services.id_fin.service import ServiceClosedError, serialize_fin_detection


def test_service_serializes_public_card_details_only() -> None:
    mrz_result = MRZResult(
        fin="1ABC234",
        confidence=0.95,
        line1="IAAZEAA12345670AZE1ABC234<<<<<",
        line2="9001011M3001019AZE<<<<<<<<<<<0",
        line3="TEST<<PERSON<<<<<<<<<<<<<<<<<<",
        checksum_valid=True,
        method="mrz_strip",
        card_type="new_card",
        card_serial_number="AA1234567",
        line_confidences=(0.95, 0.94, 0.93),
        quality_score=120,
        fin_is_canonical=True,
        secondary_checksum_valid=True,
    )

    serialized = serialize_fin_detection(
        FINDetectionOutput(
            fin="1ABC234",
            confidence=0.95,
            mrz_result=mrz_result,
        )
    )

    mrz_details = serialized["mrz_details"]
    assert isinstance(mrz_details, dict)
    assert mrz_details["card_serial_number"] == "AA1234567"
    assert {
        "line_confidences",
        "quality_score",
        "fin_is_canonical",
        "secondary_checksum_valid",
    }.isdisjoint(mrz_details)


def test_id_fin_service_processes_batch_sequentially(monkeypatch) -> None:
    from services.id_fin import service as service_module

    invocation_order: list[str] = []

    class FakeSettings:
        use_gpu = False
        save_ocr_debug_images = False

    class FakeDetector:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def detect_from_mrz(self, image_path: Path) -> FINDetectionOutput:
            invocation_order.append(image_path.name)
            return FINDetectionOutput(
                fin=f"FIN-{image_path.name}",
                confidence=0.91,
                mrz_result=None,
                notes=[],
            )

    monkeypatch.setattr(service_module, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(service_module, "FINDetector", FakeDetector)
    service = service_module.IDFinService()
    results = asyncio.run(
        service.process_many(
            image_paths=[Path("a.png"), Path("b.png"), Path("c.png")]
        )
    )

    assert [item["fin"] for item in results] == [
        "FIN-a.png",
        "FIN-b.png",
        "FIN-c.png",
    ]
    assert invocation_order == ["a.png", "b.png", "c.png"]
    service.close()


def test_sequential_batch_stops_after_processing_error(monkeypatch) -> None:
    from services.id_fin import service as service_module

    completed: list[str] = []

    class FakeSettings:
        use_gpu = False
        save_ocr_debug_images = False

    class MixedDetector:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def detect_from_mrz(self, image_path: Path) -> FINDetectionOutput:
            if image_path.name == "bad.png":
                raise OCRProcessingError("failed")
            completed.append(image_path.name)
            return FINDetectionOutput(fin="OK", confidence=0.9)

    monkeypatch.setattr(service_module, "FINDetector", MixedDetector)
    service = service_module.IDFinService(settings=FakeSettings())

    with pytest.raises(OCRProcessingError, match="failed"):
        asyncio.run(
            service.process_many(
                image_paths=[Path("bad.png"), Path("slow.png")]
            )
        )

    assert completed == []
    service.close()


def test_service_close_is_idempotent_and_rejects_new_work(monkeypatch) -> None:
    from services.id_fin import service as service_module

    class FakeSettings:
        use_gpu = False
        save_ocr_debug_images = False

    class FakeDetector:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def detect_from_mrz(self, image_path: Path) -> FINDetectionOutput:
            return FINDetectionOutput(fin=image_path.stem, confidence=0.9)

    monkeypatch.setattr(service_module, "FINDetector", FakeDetector)
    service = service_module.IDFinService(settings=FakeSettings())

    service.close()
    service.close()

    with pytest.raises(ServiceClosedError, match="closed"):
        asyncio.run(service.process_detection(image_path=Path("late.png")))
    with pytest.raises(ServiceClosedError, match="closed"):
        asyncio.run(service.process_many(image_paths=[]))


def test_service_close_drains_in_flight_work(monkeypatch) -> None:
    from services.id_fin import service as service_module

    started = Event()
    release = Event()

    class FakeSettings:
        use_gpu = False
        save_ocr_debug_images = False

    class BlockingDetector:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def detect_from_mrz(self, image_path: Path) -> FINDetectionOutput:
            started.set()
            release.wait(timeout=2)
            return FINDetectionOutput(fin=image_path.stem, confidence=0.9)

    monkeypatch.setattr(service_module, "FINDetector", BlockingDetector)
    service = service_module.IDFinService(settings=FakeSettings())

    async def run_scenario() -> FINDetectionOutput:
        processing = asyncio.create_task(
            service.process_detection(image_path=Path("active.png"))
        )
        assert await asyncio.to_thread(started.wait, 1)
        closing = asyncio.create_task(asyncio.to_thread(service.close))
        await asyncio.sleep(0.02)
        assert not closing.done()
        release.set()
        result = await processing
        await closing
        return result

    result = asyncio.run(run_scenario())
    assert result.fin == "active"
