from dataclasses import dataclass
from importlib import import_module
import logging
import os
from pathlib import Path
from typing import Callable

from . import FINDetectionOutput
from .image_utils import draw_mrz_debug, save_debug_image
from .mrz_extractor import MRZExtractor
from .preprocessor import ImagePreprocessor


logger = logging.getLogger(__name__)
_DLL_DIRECTORY_HANDLES: list[object] = []


class OCRProcessingError(RuntimeError):
    """Raised when the OCR engine cannot complete an image."""


@dataclass(frozen=True)
class OCRAttempt:
    name: str
    image_factory: Callable[[], object | None]
    is_cropped: bool


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
    def __init__(
        self,
        use_gpu: bool = True,
        debug: bool = False,
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
            "Initializing FINDetector (use_gpu=%s, debug=%s)", use_gpu, debug
        )
        ocr_engine = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            show_log=False,
            use_gpu=use_gpu,
            max_text_length=40,
            det_limit_side_len=det_limit_side_len,
            det_limit_type="max",
        )
        self.preprocessor = ImagePreprocessor()
        self.mrz_extractor = MRZExtractor(ocr_engine)
        self.debug = debug
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
            mrz_result = None
            for attempt in attempts:
                attempt_image = attempt.image_factory()
                if attempt_image is None:
                    continue
                attempted_result = self.mrz_extractor.extract(
                    attempt_image,
                    is_cropped=attempt.is_cropped,
                    attempt=attempt.name,
                )
                attempted_results.append(attempted_result)
                if self.mrz_extractor.is_structurally_valid(
                    attempted_result
                ):
                    mrz_result = attempted_result
                    notes.append(
                        f"FIN extracted using {attempt.name}."
                    )
                    break
            else:
                if attempted_results:
                    mrz_result = max(
                        attempted_results,
                        key=lambda result: result.confidence,
                    )
                notes.append(
                    "MRZ region was not reliably detected, or OCR output "
                    "did not contain a valid TD1/TD2 structure."
                )

            if mrz_result is None:
                raise RuntimeError("OCR attempt pipeline produced no result.")

            if self.debug:
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

    def _build_attempts(self, rectified) -> list[OCRAttempt]:
        deskewed_cache: dict[str, object] = {}

        def prepare_crop(image, *, binarize: bool = False):
            return self.preprocessor.bound_ocr_input(
                self.preprocessor.prepare_mrz_for_ocr(
                    image,
                    binarize=binarize,
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

        def get_strip(ratio: float, *, binarize: bool = False, source=None):
            base = rectified if source is None else source
            strip = self.preprocessor.crop_mrz_strip(base, height_ratio=ratio)
            return prepare_crop(strip, binarize=binarize)

        def get_deskewed_strip(ratio: float, *, binarize: bool = False):
            deskewed = get_deskewed()
            if deskewed is None:
                return None
            return get_strip(ratio, binarize=binarize, source=deskewed)

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
                is_cropped=True,
            ),
            OCRAttempt(
                name="mrz_strip_binarized",
                image_factory=lambda: get_strip(
                    self.preprocessor.MRZ_HEIGHT_RATIO,
                    binarize=True,
                ),
                is_cropped=True,
            ),
            OCRAttempt(
                name="mrz_roi",
                image_factory=lambda: get_mrz_roi(rectified),
                is_cropped=True,
            ),
            OCRAttempt(
                name="mrz_roi_binarized",
                image_factory=lambda: get_mrz_roi(rectified, binarize=True),
                is_cropped=True,
            ),
            OCRAttempt(
                name="mrz_strip_wide",
                image_factory=lambda: get_strip(
                    self.preprocessor.MRZ_HEIGHT_RATIO_WIDE
                ),
                is_cropped=True,
            ),
            OCRAttempt(
                name="mrz_strip_tight",
                image_factory=lambda: get_strip(
                    self.preprocessor.MRZ_HEIGHT_RATIO_TIGHT
                ),
                is_cropped=True,
            ),
            OCRAttempt(
                name="deskewed_mrz_roi",
                image_factory=lambda: get_deskewed_mrz_roi(),
                is_cropped=True,
            ),
            OCRAttempt(
                name="deskewed_mrz_strip",
                image_factory=lambda: get_deskewed_strip(
                    self.preprocessor.MRZ_HEIGHT_RATIO
                ),
                is_cropped=True,
            ),
            OCRAttempt(
                name="deskewed_mrz_strip_binarized",
                image_factory=lambda: get_deskewed_strip(
                    self.preprocessor.MRZ_HEIGHT_RATIO,
                    binarize=True,
                ),
                is_cropped=True,
            ),
            OCRAttempt(
                name="full_image",
                image_factory=lambda: self.preprocessor.bound_ocr_input(
                    rectified,
                    self.max_ocr_side,
                ),
                is_cropped=False,
            ),
            OCRAttempt(
                name="deskewed_full_image",
                image_factory=get_deskewed_full,
                is_cropped=False,
            ),
        ]
