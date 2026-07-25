from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
from functools import lru_cache
import asyncio
import logging
from pathlib import Path
from queue import Queue
from typing import Iterator

from shared.config import get_settings
from services.id_fin import FINDetectionOutput
from services.id_fin.detector import FINDetector


logger = logging.getLogger(__name__)


def serialize_fin_detection(
    result: FINDetectionOutput,
) -> dict[str, object]:
    data = asdict(result)
    data["confidence"] = round(float(result.confidence), 4)
    data["mrz_details"] = data.pop("mrz_result")
    return data


class DetectorPool:
    """Fixed pool of GPU OCR engines for bounded parallel inference."""

    def __init__(self, size: int, factory) -> None:
        if size < 1:
            raise ValueError("Detector pool size must be at least 1.")
        self.size = size
        self._queue: Queue[FINDetector] = Queue(maxsize=size)
        for index in range(size):
            logger.info("Warming OCR worker %s/%s", index + 1, size)
            self._queue.put(factory())

    @contextmanager
    def acquire(self) -> Iterator[FINDetector]:
        detector = self._queue.get()
        try:
            yield detector
        finally:
            self._queue.put(detector)

    def stats(self) -> dict[str, int]:
        return {
            "workers": self.size,
            "available": self._queue.qsize(),
        }


class IDFinService:
    """Async OCR service with a bounded GPU worker pool.

    Concurrent HTTP requests and batch items share the same pool. Up to
    ``ocr_max_concurrency`` images run at once; additional work waits for a
    free worker instead of overloading GPU memory.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.max_concurrency = settings.ocr_max_concurrency
        self._pool = DetectorPool(
            self.max_concurrency,
            lambda: FINDetector(
                use_gpu=settings.use_gpu,
                debug=settings.debug,
            ),
        )
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_concurrency,
            thread_name_prefix="id-fin-ocr",
        )
        logger.info(
            "ID FIN service ready with %s concurrent GPU workers",
            self.max_concurrency,
        )

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
        if not image_paths:
            return []

        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(
                self._executor,
                self._detect_one,
                image_path,
            )
            for image_path in image_paths
        ]
        return list(await asyncio.gather(*tasks))

    def _detect_one(self, image_path: Path) -> dict[str, object]:
        with self._pool.acquire() as detector:
            return serialize_fin_detection(
                detector.detect_from_mrz(image_path)
            )

    def concurrency_info(self) -> dict[str, object]:
        pool_stats = self._pool.stats()
        return {
            "max_concurrency": self.max_concurrency,
            "workers": pool_stats["workers"],
            "available_workers": pool_stats["available"],
        }


@lru_cache(maxsize=1)
def get_id_fin_service() -> IDFinService:
    return IDFinService()
