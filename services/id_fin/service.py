from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
from functools import lru_cache
import asyncio
import logging
from pathlib import Path
from queue import Queue
from typing import Callable, Iterator

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
    return data


class DetectorPool:
    """Fixed pool of GPU OCR engines for bounded parallel inference."""

    def __init__(
        self,
        size: int,
        factory: Callable[[], FINDetector],
    ) -> None:
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

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        use_gpu: bool | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        settings = settings or get_settings()
        self.max_concurrency = (
            max_concurrency
            if max_concurrency is not None
            else settings.ocr_max_concurrency
        )
        detector_uses_gpu = settings.use_gpu if use_gpu is None else use_gpu
        self._pool = DetectorPool(
            self.max_concurrency,
            lambda: FINDetector(
                use_gpu=detector_uses_gpu,
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
            self._detect_one,
            image_path,
        )

    async def process_many(
        self,
        *,
        image_paths: list[Path],
    ) -> list[dict[str, object]]:
        if not image_paths:
            return []

        tasks = [
            self.process_detection(image_path=image_path)
            for image_path in image_paths
        ]
        return [
            serialize_fin_detection(result)
            for result in await asyncio.gather(*tasks)
        ]

    def _detect_one(self, image_path: Path) -> FINDetectionOutput:
        with self._pool.acquire() as detector:
            return detector.detect_from_mrz(image_path)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

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
