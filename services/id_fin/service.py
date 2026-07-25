from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import lru_cache
import asyncio
import logging
from pathlib import Path

from shared.config import Settings, get_settings
from services.id_fin import FINDetectionOutput
from services.id_fin.detector import FINDetector


logger = logging.getLogger(__name__)


def serialize_fin_detection(
    result: FINDetectionOutput,
) -> dict[str, object]:
    data = asdict(result)
    data["confidence"] = round(float(result.confidence), 4)
    data["mrz_details"] = data.pop("mrz_result")
    if data["mrz_details"] is not None:
        for internal_field in (
            "line_confidences",
            "quality_score",
            "fin_is_canonical",
            "secondary_checksum_valid",
        ):
            data["mrz_details"].pop(internal_field, None)
    return data


class IDFinService:
    """Async adapter around one stable PaddleOCR engine."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        use_gpu: bool | None = None,
    ) -> None:
        settings = settings or get_settings()
        detector_uses_gpu = settings.use_gpu if use_gpu is None else use_gpu
        self._detector = FINDetector(
            use_gpu=detector_uses_gpu,
            save_debug_images=settings.save_ocr_debug_images,
        )
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="id-fin-ocr",
        )
        logger.info("ID FIN service ready with one OCR engine")

    async def process(
        self,
        *,
        image_path: Path,
    ) -> dict[str, object]:
        result = await self.process_detection(image_path=image_path)
        return serialize_fin_detection(result)

    async def process_detection(
        self,
        *,
        image_path: Path,
    ) -> FINDetectionOutput:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._detector.detect_from_mrz,
            image_path,
        )

    async def process_many(
        self,
        *,
        image_paths: list[Path],
    ) -> list[dict[str, object]]:
        if not image_paths:
            return []

        results = []
        for image_path in image_paths:
            result = await self.process_detection(image_path=image_path)
            results.append(serialize_fin_detection(result))
        return results

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

@lru_cache(maxsize=1)
def get_id_fin_service() -> IDFinService:
    return IDFinService()
