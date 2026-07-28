from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib import import_module
import logging
import os
from pathlib import Path
import time
from typing import Callable

from . import FINDetectionOutput
from .image_utils import draw_mrz_debug, save_debug_image
from .mrz_extractor import MRZExtractor
from .preprocessor import ImagePreprocessor


logger = logging.getLogger(__name__)
_DLL_DIRECTORY_HANDLES: list[object] = []
_NVIDIA_LIBS_CONFIGURED = False
PRODUCTION_OCR_VERSION = "PP-OCRv3"
_NVIDIA_CUDA_MODULES = (
    "nvidia.cudnn",
    "nvidia.cublas",
    "nvidia.cuda_nvrtc",
)


class OCRProcessingError(RuntimeError):
    """Raised when the OCR engine cannot complete an image."""


@dataclass(frozen=True)
class OCRAttempt:
    name: str
    image_factory: Callable[[], object | None]


def _running_on_windows() -> bool:
    return os.name == "nt"


def _nvidia_package_lib_directories(module_name: str) -> list[str]:
    """Return bin/lib folders shipped by a pip nvidia-* package, if present."""
    module = import_module(module_name)
    package_root = Path(next(iter(module.__path__)))
    directories: list[str] = []
    # Windows wheels keep shared libs under bin/; Linux wheels under lib/.
    for subdir_name in ("bin", "lib"):
        candidate = package_root / subdir_name
        if candidate.is_dir():
            directories.append(str(candidate.resolve()))
    return directories


def configure_nvidia_dll_directories() -> None:
    """Expose pip-installed NVIDIA CUDA libs to Paddle on Windows and Linux.

    Windows uses ``os.add_dll_directory`` + ``PATH``. Linux prepends package
    ``lib``/``bin`` folders to ``LD_LIBRARY_PATH`` before Paddle loads cuDNN.
    """
    global _NVIDIA_LIBS_CONFIGURED
    if _NVIDIA_LIBS_CONFIGURED:
        return
    _NVIDIA_LIBS_CONFIGURED = True

    directories: list[str] = []
    for module_name in _NVIDIA_CUDA_MODULES:
        try:
            directories.extend(_nvidia_package_lib_directories(module_name))
        except ImportError:
            logger.warning(
                "Optional NVIDIA package %s is not installed; "
                "GPU OCR may fail to load cuDNN/CUDA shared libraries. "
                "On Linux RTX 50-series hosts install the cu12 packages from "
                "requirements-gpu-linux-5090.txt.",
                module_name,
            )

    if not directories:
        return

    if _running_on_windows():
        for directory in directories:
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(directory))
        os.environ["PATH"] = os.pathsep.join(
            [*directories, os.environ.get("PATH", "")]
        )
        return

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
        [*directories, *([existing] if existing else [])]
    )
    logger.info(
        "Prepended NVIDIA library directories to LD_LIBRARY_PATH: %s",
        directories,
    )


# Paddle reads these once, at import time, so they must be set before the
# first `import paddle` anywhere in the process.
PADDLE_ALLOCATOR_DEFAULTS = {
    # Grow the GPU pool on demand instead of Paddle's default attempt to
    # reserve ~92% of the card up front. On a machine whose GPU also drives
    # the display there may only be a couple of spare GiB, and the up-front
    # reservation leaves the pool thrashing on every new allocation.
    "FLAGS_allocator_strategy": "auto_growth",
    # Grow in 1 GiB steps. Each growth is a blocking cudaMalloc that costs
    # roughly a second under memory pressure, so fewer, larger steps beat
    # many small ones.
    "FLAGS_initial_gpu_memory_in_mb": "1024",
    "FLAGS_reallocate_gpu_memory_in_mb": "1024",
}


def configure_paddle_allocator() -> None:
    """Apply GPU allocator defaults, leaving any operator override intact."""
    for flag, value in PADDLE_ALLOCATOR_DEFAULTS.items():
        os.environ.setdefault(flag, value)


class FINDetector:
    HIGH_CONFIDENCE_ACCEPT = 0.90
    RECTIFIED_FAIL_FAST_ATTEMPTS = 3
    ORIGINAL_IMAGE_RECOVERY_ATTEMPTS = frozenset(
        {"mrz_strip", "mrz_roi", "full_image"}
    )
    RECOVERY_ATTEMPTS = frozenset(
        {
            "deskewed_mrz_roi",
            "deskewed_mrz_strip",
            "deskewed_mrz_strip_binarized",
            "full_image",
            "deskewed_full_image",
            "line_recognition",
            "original_mrz_strip",
            "original_mrz_roi",
            "original_full_image",
        }
    )
    def __init__(
        self,
        use_gpu: bool = True,
        save_debug_images: bool = False,
        max_ocr_side: int = ImagePreprocessor.DEFAULT_MAX_OCR_SIDE,
        det_limit_side_len: int = 736,
        concurrent_attempts: int = 1,
    ):
        if use_gpu:
            configure_nvidia_dll_directories()
            configure_paddle_allocator()

        import paddle
        from paddleocr import PaddleOCR

        if use_gpu and not paddle.device.is_compiled_with_cuda():
            raise RuntimeError(
                "GPU mode requires the paddlepaddle-gpu package. "
                "On Windows use requirements.txt. On Linux RTX 50-series "
                "(Blackwell) use requirements-gpu-linux-5090.txt with "
                "paddlepaddle-gpu 3.2.1+ from the cu129 index. "
                "Or set USE_GPU=false for CPU OCR."
            )

        logger.info(
            "Initializing FINDetector (use_gpu=%s, save_debug_images=%s, "
            "concurrent_attempts=%d)",
            use_gpu,
            save_debug_images,
            concurrent_attempts,
        )

        def build_engine() -> "PaddleOCR":
            try:
                return PaddleOCR(
                    use_angle_cls=False,
                    lang="en",
                    show_log=False,
                    use_gpu=use_gpu,
                    max_text_length=40,
                    det_limit_side_len=det_limit_side_len,
                    det_limit_type="max",
                    ocr_version=PRODUCTION_OCR_VERSION,
                )
            except Exception as exc:
                if use_gpu:
                    raise RuntimeError(
                        "Failed to initialize PaddleOCR with GPU. "
                        "Missing/incompatible CUDA or cuDNN is a common "
                        "cause on Linux (especially RTX 50-series). "
                        "Install requirements-gpu-linux-5090.txt, confirm "
                        "LD_LIBRARY_PATH includes nvidia-* lib folders, or "
                        "set USE_GPU=false. "
                        f"Original error: {exc}"
                    ) from exc
                raise

        self.preprocessor = ImagePreprocessor()
        try:
            self.mrz_extractor = MRZExtractor(build_engine())
        except RuntimeError:
            raise
        except Exception as exc:
            if use_gpu and (
                "cudnn" in str(exc).lower() or "cuda" in str(exc).lower()
            ):
                raise RuntimeError(
                    "GPU OCR backend failed during engine setup "
                    f"({exc}). Set USE_GPU=false for CPU, or install the "
                    "Linux RTX 50-series stack from "
                    "requirements-gpu-linux-5090.txt."
                ) from exc
            raise
        # Extra engines let independent attempts within a single image run
        # concurrently instead of strictly one at a time. This trades some
        # redundant GPU/CPU compute (an attempt may still run even though an
        # earlier concurrent one already made it unnecessary) for lower
        # wall-clock detection latency. Defaults to 1 engine, which keeps
        # the original strictly-sequential behavior unchanged.
        extra_engine_count = max(0, concurrent_attempts - 1)
        self._extractor_pool = [self.mrz_extractor] + [
            MRZExtractor(build_engine()) for _ in range(extra_engine_count)
        ]
        self._attempt_executor = (
            ThreadPoolExecutor(
                max_workers=len(self._extractor_pool),
                thread_name_prefix="id-fin-attempt",
            )
            if extra_engine_count
            else None
        )
        self.save_debug_images = save_debug_images
        self.max_ocr_side = max_ocr_side

    def detect_from_mrz(self, image_path: str | Path) -> FINDetectionOutput:
        image_path = Path(image_path)
        notes: list[str] = []
        try:
            image = self.preprocessor.load(image_path)
            rectified = self.preprocessor.detect_card_roi(image)
            notes.append(
                "Card ROI boundary not detected; using original image."
                if rectified is image
                else "Card ROI successfully detected and rectified."
            )
            attempts = self._build_attempts(rectified)
            attempted_results = []
            valid_results = []
            roi_applied = rectified is not image

            self._run_attempt_cascade(
                attempts,
                attempted_results=attempted_results,
                valid_results=valid_results,
                notes=notes,
                image_path=image_path,
                roi_applied=roi_applied,
            )

            if not valid_results and roi_applied:
                for attempt in self._build_attempts(image):
                    if (
                        attempt.name
                        not in self.ORIGINAL_IMAGE_RECOVERY_ATTEMPTS
                    ):
                        continue
                    attempt_image = attempt.image_factory()
                    if attempt_image is None:
                        continue
                    attempt_name = f"original_{attempt.name}"
                    attempt_started = time.perf_counter()
                    attempted_result = self.mrz_extractor.extract(
                        attempt_image,
                        attempt=attempt_name,
                    )
                    attempt_seconds = time.perf_counter() - attempt_started
                    notes.append(
                        f"OCR attempt {attempt_name}: "
                        f"{attempt_seconds:.4f}s."
                    )
                    attempted_results.append(attempted_result)
                    if not self.mrz_extractor.is_structurally_valid(
                        attempted_result
                    ):
                        continue
                    valid_results.append(attempted_result)
                    if self._should_stop_after_valid_result(
                        attempted_result,
                        valid_results,
                        notes,
                        attempt_name=attempt_name,
                    ):
                        break

            if not valid_results:
                line_images = self.preprocessor.extract_td2_line_crops(
                    rectified
                )
                if len(line_images) == 2:
                    attempt_started = time.perf_counter()
                    attempted_result = (
                        self.mrz_extractor.extract_recognition_lines(
                            line_images,
                            attempt="line_recognition",
                        )
                    )
                    attempt_seconds = time.perf_counter() - attempt_started
                    notes.append(
                        "OCR attempt line_recognition: "
                        f"{attempt_seconds:.4f}s."
                    )
                    attempted_results.append(attempted_result)
                    if self.mrz_extractor.is_structurally_valid(
                        attempted_result
                    ):
                        valid_results.append(attempted_result)

            return self._finalize_detection(
                rectified=rectified,
                attempted_results=attempted_results,
                valid_results=valid_results,
                notes=notes,
                image_path=image_path,
            )
        except Exception as exc:
            logger.exception("Error processing MRZ image %s", image_path.name)
            raise OCRProcessingError(
                f"Failed to process MRZ image {image_path.name}."
            ) from exc

    def _run_attempt_cascade(
        self,
        attempts: list[OCRAttempt],
        *,
        attempted_results: list,
        valid_results: list,
        notes: list[str],
        image_path: Path,
        roi_applied: bool,
    ) -> None:
        """Runs the primary attempt cascade, stopping early on consensus or
        high confidence exactly as a single-attempt-at-a-time loop would.

        When extra OCR engines are configured (``concurrent_attempts`` on
        __init__), independent attempts run concurrently in small batches
        instead of strictly one at a time: each batch is dispatched using
        the consensus/validity state as of the start of that batch, so an
        attempt may occasionally run even though an earlier attempt in the
        same batch already made it unnecessary. That's an intentional
        trade of some redundant compute for lower wall-clock latency; at
        the default of one engine, batches are always size 1 and behavior
        is identical to the original sequential loop.
        """
        extractor_pool = getattr(self, "_extractor_pool", None) or [
            self.mrz_extractor
        ]
        executor = getattr(self, "_attempt_executor", None)
        batch_size = len(extractor_pool) if executor is not None else 1

        failed_rectified_attempts = 0
        index = 0
        total = len(attempts)

        while index < total:
            fin_counts = Counter(
                result.fin for result in valid_results if result.fin
            )
            has_conflict = len(fin_counts) > 1
            strongest_consensus = max(fin_counts.values(), default=0)

            batch: list[OCRAttempt] = []
            cascade_done = False
            while len(batch) < batch_size and index < total:
                attempt = attempts[index]
                if (
                    attempt.name in self.RECOVERY_ATTEMPTS
                    and strongest_consensus >= 2
                    and not has_conflict
                ):
                    cascade_done = True
                    break
                index += 1
                batch.append(attempt)

            prepared = [
                (attempt, attempt.image_factory()) for attempt in batch
            ]
            prepared = [
                (attempt, attempt_image)
                for attempt, attempt_image in prepared
                if attempt_image is not None
            ]

            if not prepared:
                if cascade_done:
                    return
                continue

            if len(prepared) == 1 or executor is None:
                timed_results = [
                    self._time_attempt(extractor_pool[0], attempt, attempt_image)
                    for attempt, attempt_image in prepared
                ]
            else:
                futures = [
                    executor.submit(
                        self._time_attempt,
                        extractor_pool[position % len(extractor_pool)],
                        attempt,
                        attempt_image,
                    )
                    for position, (attempt, attempt_image) in enumerate(
                        prepared
                    )
                ]
                timed_results = [future.result() for future in futures]

            for attempt, attempted_result, attempt_seconds in timed_results:
                notes.append(
                    f"OCR attempt {attempt.name}: "
                    f"{attempt_seconds:.4f}s."
                )
                logger.info(
                    "OCR attempt %s for %s completed in %.4fs",
                    attempt.name,
                    image_path.name,
                    attempt_seconds,
                )
                attempted_results.append(attempted_result)
                if self.mrz_extractor.is_structurally_valid(
                    attempted_result
                ):
                    valid_results.append(attempted_result)
                    if self._should_stop_after_valid_result(
                        attempted_result,
                        valid_results,
                        notes,
                        attempt_name=attempt.name,
                    ):
                        return
                elif roi_applied and not valid_results:
                    failed_rectified_attempts += 1
                    if (
                        failed_rectified_attempts
                        >= self.RECTIFIED_FAIL_FAST_ATTEMPTS
                    ):
                        notes.append(
                            "Rectified crop produced no MRZ after "
                            f"{self.RECTIFIED_FAIL_FAST_ATTEMPTS} attempts; "
                            "trying original image."
                        )
                        return

            if cascade_done:
                return

    @staticmethod
    def _time_attempt(
        extractor,
        attempt: OCRAttempt,
        attempt_image,
    ) -> tuple[OCRAttempt, object, float]:
        attempt_started = time.perf_counter()
        result = extractor.extract(attempt_image, attempt=attempt.name)
        return attempt, result, time.perf_counter() - attempt_started

    def _finalize_detection(
        self,
        *,
        rectified,
        attempted_results: list,
        valid_results: list,
        notes: list[str],
        image_path: Path,
    ) -> FINDetectionOutput:
        if valid_results:
            fin_counts = Counter(
                result.fin for result in valid_results if result.fin
            )
            mrz_result = max(
                valid_results,
                key=lambda result: self._result_rank(
                    result,
                    fin_counts,
                ),
            )
            notes.append(
                f"FIN selected from {mrz_result.method} with "
                f"{fin_counts[mrz_result.fin]} agreeing attempt(s)."
            )
        elif attempted_results:
            mrz_result = max(
                attempted_results,
                key=lambda result: result.confidence,
            )
            notes.append(
                "MRZ region was not reliably detected, or OCR output "
                "did not contain a valid TD1/TD2 structure."
            )
        else:
            mrz_result = None

        if mrz_result is None:
            raise RuntimeError("OCR attempt pipeline produced no result.")

        if self.save_debug_images:
            annotated = draw_mrz_debug(
                rectified,
                mrz_result.line1,
                mrz_result.line2,
                mrz_result.line3,
                mrz_result.fin or "NOT_FOUND",
            )
            save_debug_image(annotated, f"debug_mrz_{image_path.name}")
            notes.append("Saved MRZ debug image.")
        return FINDetectionOutput(
            fin=mrz_result.fin,
            confidence=mrz_result.confidence,
            mrz_result=mrz_result,
            notes=notes,
        )

    @classmethod
    def _is_high_confidence_accept(cls, result) -> bool:
        return (
            bool(result.checksum_valid)
            and bool(result.fin_is_canonical)
            and float(result.confidence) >= cls.HIGH_CONFIDENCE_ACCEPT
        )

    @classmethod
    def _should_stop_after_valid_result(
        cls,
        attempted_result,
        valid_results: list,
        notes: list[str],
        *,
        attempt_name: str,
    ) -> bool:
        agreement = sum(
            result.fin == attempted_result.fin for result in valid_results
        )
        distinct_fins = {
            result.fin for result in valid_results if result.fin
        }
        if len(distinct_fins) > 1:
            return False
        if cls._is_high_confidence_accept(attempted_result):
            notes.append(
                f"High-confidence FIN accepted after {attempt_name}."
            )
            return True
        if agreement >= 2:
            notes.append(
                f"Strong FIN consensus reached after {attempt_name}."
            )
            return True
        return False

    @staticmethod
    def _result_rank(result, fin_counts: Counter) -> tuple:
        return (
            fin_counts[result.fin],
            int(result.checksum_valid)
            + int(result.secondary_checksum_valid),
            int(result.fin_is_canonical),
            result.quality_score,
            result.confidence,
        )

    def _build_attempts(self, rectified) -> list[OCRAttempt]:
        deskewed_cache: dict[str, object] = {}
        mrz_roi_cache: dict[int, object] = {}
        deskewed_strip_cache: dict[float, object] = {}

        def prepare_crop(
            image,
            *,
            binarize: bool = False,
            adaptive_upscale: bool = False,
        ):
            return self.preprocessor.bound_ocr_input(
                self.preprocessor.prepare_mrz_for_ocr(
                    image,
                    binarize=binarize,
                    adaptive_upscale=adaptive_upscale,
                ),
                self.max_ocr_side,
            )

        def get_mrz_roi(source, *, binarize: bool = False):
            # Keyed by identity: the same source (rectified or deskewed) is
            # passed in twice across the plain/binarized attempt pair, and
            # ROI detection (contours/Sobel/morphology) is expensive to redo.
            source_key = id(source)
            if source_key not in mrz_roi_cache:
                mrz_roi_cache[source_key] = self.preprocessor.detect_mrz_roi(
                    source
                )
            mrz_roi = mrz_roi_cache[source_key]
            if mrz_roi is None:
                return None
            return prepare_crop(mrz_roi, binarize=binarize)

        def get_deskewed():
            if "image" not in deskewed_cache:
                deskewed_cache["image"] = self.preprocessor.deskew(rectified)
            deskewed = deskewed_cache["image"]
            return None if deskewed is rectified else deskewed

        def get_strip(ratio: float, *, binarize: bool = False):
            strip = self.preprocessor.crop_mrz_strip(
                rectified,
                height_ratio=ratio,
            )
            return prepare_crop(strip, binarize=binarize)

        def get_deskewed_strip(ratio: float, *, binarize: bool = False):
            # Same rationale as get_mrz_roi: the plain/binarized attempt
            # pair shares the same Hough-transform deskew of one strip crop.
            if ratio not in deskewed_strip_cache:
                strip = self.preprocessor.crop_mrz_strip(
                    rectified,
                    height_ratio=ratio,
                )
                deskewed = self.preprocessor.deskew_mrz_strip(strip)
                deskewed_strip_cache[ratio] = (
                    None if deskewed is strip else deskewed
                )
            deskewed = deskewed_strip_cache[ratio]
            if deskewed is None:
                return None
            return prepare_crop(
                deskewed,
                binarize=binarize,
                adaptive_upscale=True,
            )

        def get_deskewed_mrz_roi(*, binarize: bool = False):
            deskewed = get_deskewed()
            return None if deskewed is None else get_mrz_roi(deskewed, binarize=binarize)

        def get_deskewed_full():
            deskewed = get_deskewed()
            if deskewed is None:
                return None
            return self.preprocessor.bound_ocr_input(
                deskewed,
                self.max_ocr_side,
            )

        # Ordered by measured yield-per-cost on the benchmark fixtures and on
        # recorded production traffic, cheapest high-yield first:
        #   - the plain MRZ strip resolves the large majority of images alone;
        #   - full_image is the single highest-yield fallback, so it runs
        #     early rather than after every crop variant has been tried. It is
        #     only expensive when the GPU pool is thrashing, which
        #     configure_paddle_allocator() addresses;
        #   - mrz_roi and the binarized/deskewed variants are low-yield
        #     rescues for awkward crops, so they run last.
        return [
            OCRAttempt(
                name="mrz_strip",
                image_factory=lambda: get_strip(
                    self.preprocessor.MRZ_HEIGHT_RATIO
                ),
            ),
            OCRAttempt(
                name="mrz_strip_wide",
                image_factory=lambda: get_strip(
                    self.preprocessor.MRZ_HEIGHT_RATIO_WIDE
                ),
            ),
            OCRAttempt(
                name="full_image",
                image_factory=lambda: self.preprocessor.bound_ocr_input(
                    rectified,
                    self.max_ocr_side,
                ),
            ),
            OCRAttempt(
                name="mrz_strip_tight",
                image_factory=lambda: get_strip(
                    self.preprocessor.MRZ_HEIGHT_RATIO_TIGHT
                ),
            ),
            OCRAttempt(
                name="mrz_roi",
                image_factory=lambda: get_mrz_roi(rectified),
            ),
            OCRAttempt(
                name="mrz_strip_binarized",
                image_factory=lambda: get_strip(
                    self.preprocessor.MRZ_HEIGHT_RATIO,
                    binarize=True,
                ),
            ),
            OCRAttempt(
                name="mrz_roi_binarized",
                image_factory=lambda: get_mrz_roi(rectified, binarize=True),
            ),
            OCRAttempt(
                name="deskewed_mrz_strip",
                image_factory=lambda: get_deskewed_strip(
                    self.preprocessor.MRZ_HEIGHT_RATIO
                ),
            ),
            OCRAttempt(
                name="deskewed_mrz_strip_binarized",
                image_factory=lambda: get_deskewed_strip(
                    self.preprocessor.MRZ_HEIGHT_RATIO,
                    binarize=True,
                ),
            ),
            OCRAttempt(
                name="deskewed_mrz_roi",
                image_factory=lambda: get_deskewed_mrz_roi(),
            ),
            OCRAttempt(
                name="deskewed_full_image",
                image_factory=get_deskewed_full,
            ),
        ]
