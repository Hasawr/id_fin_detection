from importlib import import_module
import logging
import os
from pathlib import Path

from . import FINDetectionOutput
from .image_utils import draw_mrz_debug, save_debug_image
from .mrz_extractor import MRZExtractor
from .preprocessor import ImagePreprocessor


logger = logging.getLogger(__name__)
_DLL_DIRECTORY_HANDLES: list[object] = []


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
    def __init__(self, use_gpu: bool = True, debug: bool = False):
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
        )
        self.preprocessor = ImagePreprocessor()
        self.mrz_extractor = MRZExtractor(ocr_engine)
        self.debug = debug

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
            enhanced = self.preprocessor.enhance_for_mrz(rectified)
            mrz_result = self.mrz_extractor.extract(enhanced, is_cropped=True)
            if not mrz_result.fin or len(mrz_result.fin) != 7:
                mrz_result = self.mrz_extractor.extract(
                    rectified, is_cropped=False
                )
                notes.append("FIN extraction fell back to full image scan.")
            else:
                notes.append("FIN successfully extracted from cropped MRZ band.")

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
        except Exception:
            logger.exception("Error processing MRZ image %s", image_path.name)
            return FINDetectionOutput(
                fin=None,
                confidence=0.0,
                notes=["Failed to process MRZ image."],
            )
