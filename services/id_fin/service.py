from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from threading import Lock

from starlette.concurrency import run_in_threadpool

from shared.config import get_settings
from services.id_fin import FINDetectionOutput
from services.id_fin.detector import FINDetector


def serialize_fin_detection(
    result: FINDetectionOutput,
) -> dict[str, object]:
    data = asdict(result)
    data["confidence"] = round(float(result.confidence), 4)
    data["mrz_details"] = data.pop("mrz_result")
    return data


class IDFinService:
    def __init__(self) -> None:
        settings = get_settings()
        self._detector = FINDetector(
            use_gpu=settings.use_gpu,
            debug=settings.debug,
        )
        self._inference_lock = Lock()

    async def process(
        self,
        *,
        image_path: Path,
    ) -> dict[str, object]:
        results = await self.process_many(image_paths=[image_path])
        return results[0]

    async def process_many(
        self,
        *,
        image_paths: list[Path],
    ) -> list[dict[str, object]]:
        def run_detection() -> list[dict[str, object]]:
            with self._inference_lock:
                return [
                    serialize_fin_detection(
                        self._detector.detect_from_mrz(image_path)
                    )
                    for image_path in image_paths
                ]

        return await run_in_threadpool(run_detection)

@lru_cache(maxsize=1)
def get_id_fin_service() -> IDFinService:
    return IDFinService()
