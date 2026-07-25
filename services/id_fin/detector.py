from collections import Counter
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
PRODUCTION_OCR_VERSION = "PP-OCRv3"


class OCRProcessingError(RuntimeError):
    """Raised when the OCR engine cannot complete an image."""


@dataclass(frozen=True)
class OCRAttempt:
    name: str
    image_factory: Callable[[], object | None]


def configure_nvidia_dll_directories() -> None:
    """Expose pip-installed NVIDIA DLLs to Paddle on Windows."""
    if os.name != "nt" or _DLL_DIRECTORY_HANDLES:
        return

    bin_directories: list[str] = []
    for module_name in ("nvidia.cudnn", "nvidia.cublas", "nvidia.cuda_nvrtc"):
        module = import_module(module_name)
        bin_directory = Path(next(iter(module.__path__))) / "bin"
        bin_directories.append(str(bin_directory))
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(bin_directory)))

    os.environ["PATH"] = os.pathsep.join(
        [*bin_directories, os.environ.get("PATH", "")]
    )


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
        det_limit_side_len: int = 960,
    ):
        if use_gpu:
            configure_nvidia_dll_directories()

        import paddle
        from paddleocr import PaddleOCR

        if use_gpu and not paddle.device.is_compiled_with_cuda():
            raise RuntimeError(
                "GPU mode requires the paddlepaddle-gpu package. "
                "Reinstall dependencies from requirements.txt."
            )

        logger.info(
            "Initializing FINDetector (use_gpu=%s, save_debug_images=%s)",
            use_gpu,
            save_debug_images,
        )
        ocr_engine = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            show_log=False,
            use_gpu=use_gpu,
            max_text_length=40,
            det_limit_side_len=det_limit_side_len,
            det_limit_type="max",
            ocr_version=PRODUCTION_OCR_VERSION,
        )
        self.preprocessor = ImagePreprocessor()
        self.mrz_extractor = MRZExtractor(ocr_engine)
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
            failed_rectified_attempts = 0
            for attempt in attempts:
                fin_counts = Counter(
                    result.fin for result in valid_results if result.fin
                )
                has_conflict = len(fin_counts) > 1
                strongest_consensus = max(
                    fin_counts.values(),
                    default=0,
                )
                if (
                    attempt.name in self.RECOVERY_ATTEMPTS
                    and strongest_consensus >= 2
                    and not has_conflict
                ):
                    break
                attempt_image = attempt.image_factory()
                if attempt_image is None:
                    continue
                attempt_started = time.perf_counter()
                attempted_result = self.mrz_extractor.extract(
                    attempt_image,
                    attempt=attempt.name,
                )
                attempt_seconds = time.perf_counter() - attempt_started
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
                        break
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
                        break

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
        except Exception as exc:
            logger.exception("Error processing MRZ image %s", image_path.name)
            raise OCRProcessingError(
                f"Failed to process MRZ image {image_path.name}."
            ) from exc

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
            mrz_roi = self.preprocessor.detect_mrz_roi(source)
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
            strip = self.preprocessor.crop_mrz_strip(
                rectified,
                height_ratio=ratio,
            )
            deskewed = self.preprocessor.deskew_mrz_strip(strip)
            if deskewed is strip:
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

        return [
            OCRAttempt(
                name="mrz_strip",
                image_factory=lambda: get_strip(
                    self.preprocessor.MRZ_HEIGHT_RATIO
                ),
            ),
            OCRAttempt(
                name="mrz_roi",
                image_factory=lambda: get_mrz_roi(rectified),
            ),
            OCRAttempt(
                name="mrz_strip_wide",
                image_factory=lambda: get_strip(
                    self.preprocessor.MRZ_HEIGHT_RATIO_WIDE
                ),
            ),
            OCRAttempt(
                name="mrz_strip_tight",
                image_factory=lambda: get_strip(
                    self.preprocessor.MRZ_HEIGHT_RATIO_TIGHT
                ),
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
                name="deskewed_mrz_roi",
                image_factory=lambda: get_deskewed_mrz_roi(),
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
                name="full_image",
                image_factory=lambda: self.preprocessor.bound_ocr_input(
                    rectified,
                    self.max_ocr_side,
                ),
            ),
            OCRAttempt(
                name="deskewed_full_image",
                image_factory=get_deskewed_full,
            ),
        ]
