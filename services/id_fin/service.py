from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import lru_cache
import asyncio
import logging
from pathlib import Path
from threading import Event, Lock

from shared.config import Settings, get_settings
from services.id_fin import FINDetectionOutput
from services.id_fin.detector import FINDetector


logger = logging.getLogger(__name__)


class ServiceClosedError(RuntimeError):
    """Raised when OCR work is submitted after service shutdown."""


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
    """Async adapter around one or more PaddleOCR engines.

    ``settings.ocr_worker_count`` (default 1) controls how many independent
    ``FINDetector`` instances/engines run behind the executor. At the
    default of 1, behavior is unchanged: one engine, strictly sequential,
    fail-fast batch processing. Raising it trades GPU/CPU memory for
    throughput by processing images concurrently across engines.

    ``settings.ocr_concurrent_attempts`` (default 1) controls how many extra
    OCR engines *each* ``FINDetector`` holds so it can run independent
    attempts on a single image concurrently, cutting per-image detection
    latency. See ``FINDetector._run_attempt_cascade`` for the trade-off.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        use_gpu: bool | None = None,
    ) -> None:
        settings = settings or get_settings()
        detector_uses_gpu = settings.use_gpu if use_gpu is None else use_gpu
        worker_count = max(1, getattr(settings, "ocr_worker_count", 1))
        concurrent_attempts = max(
            1, getattr(settings, "ocr_concurrent_attempts", 1)
        )
        det_limit_side_len = max(
            320, getattr(settings, "ocr_det_limit_side_len", 736)
        )
        self._detectors = [
            FINDetector(
                use_gpu=detector_uses_gpu,
                save_debug_images=settings.save_ocr_debug_images,
                concurrent_attempts=concurrent_attempts,
                det_limit_side_len=det_limit_side_len,
            )
            for _ in range(worker_count)
        ]
        self._next_detector_index = 0
        self._executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="id-fin-ocr",
        )
        self._lifecycle_lock = Lock()
        self._is_closed = False
        self._close_complete = Event()
        logger.info(
            "ID FIN service ready with %d OCR engine(s)", worker_count
        )

    def _acquire_detector(self) -> FINDetector:
        """Round-robin over engines. Caller must hold ``_lifecycle_lock``."""
        detector = self._detectors[self._next_detector_index]
        self._next_detector_index = (
            self._next_detector_index + 1
        ) % len(self._detectors)
        return detector

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
        with self._lifecycle_lock:
            if self._is_closed:
                raise ServiceClosedError("ID FIN service is closed.")
            detector = self._acquire_detector()
            future = self._executor.submit(
                detector.detect_from_mrz,
                image_path,
            )
        return await asyncio.wrap_future(future)

    async def process_many(
        self,
        *,
        image_paths: list[Path],
    ) -> list[dict[str, object]]:
        with self._lifecycle_lock:
            if self._is_closed:
                raise ServiceClosedError("ID FIN service is closed.")
        if not image_paths:
            return []

        if len(self._detectors) == 1:
            # Single-engine path: strictly sequential and fail-fast, so a
            # processing error stops the batch before later images start.
            results = []
            for image_path in image_paths:
                result = await self.process_detection(image_path=image_path)
                results.append(serialize_fin_detection(result))
            return results

        detections = await asyncio.gather(
            *(
                self.process_detection(image_path=image_path)
                for image_path in image_paths
            )
        )
        return [serialize_fin_detection(result) for result in detections]

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._is_closed:
                is_first_closer = False
            else:
                self._is_closed = True
                is_first_closer = True
        if not is_first_closer:
            self._close_complete.wait()
            return
        try:
            self._executor.shutdown(wait=True, cancel_futures=False)
        finally:
            self._close_complete.set()
        logger.info("ID FIN service shut down after draining OCR work")

@lru_cache(maxsize=1)
def get_id_fin_service() -> IDFinService:
    return IDFinService()
