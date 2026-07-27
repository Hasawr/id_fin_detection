import os
import glob
import site
import ctypes
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

def _preload_cuda_libraries():
    """Pre-load NVIDIA CUDA/cuDNN shared libraries into global symbol table."""
    try:
        site_packages = site.getsitepackages()
        for sp in site_packages:
            nvidia_dirs = glob.glob(os.path.join(sp, 'nvidia', '*', 'lib'))
            if nvidia_dirs:
                existing_ld = os.environ.get('LD_LIBRARY_PATH', '')
                os.environ['LD_LIBRARY_PATH'] = ':'.join(nvidia_dirs) + (':' + existing_ld if existing_ld else '')
                for d in nvidia_dirs:
                    for so_file in sorted(glob.glob(os.path.join(d, '*.so*'))):
                        try:
                            ctypes.CDLL(so_file, mode=ctypes.RTLD_GLOBAL)
                        except Exception:
                            pass
    except Exception as e:
        logger.debug(f"NVIDIA library pre-load notice: {e}")

class OCREngineAdapter:
    """
    Unified OCR Engine Adapter.
    - CPU mode (use_gpu=False): Uses stock official PaddleOCR(lang='en') natively.
    - GPU mode (use_gpu=True): Uses CUDA 12 accelerated engine tuned to match PaddleOCR parameters.
    """

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        logger.info(f"Initializing OCREngineAdapter (use_gpu={use_gpu})...")

        if not use_gpu:
            # CPU mode: Use native official PaddleOCR with lang='en'
            try:
                from paddleocr import PaddleOCR
                self.mode = "paddleocr_cpu"
                self.engine = PaddleOCR(
                    use_angle_cls=False, 
                    lang='en', 
                    show_log=False, 
                    use_gpu=False, 
                    enable_mkldnn=False
                )
                logger.info("Native PaddleOCR CPU engine initialized successfully.")
                return
            except Exception as e:
                logger.warning(f"Native PaddleOCR CPU initialization failed: {e}. Falling back to RapidOCR CPU.")

        # GPU mode (or CPU fallback if paddleocr failed)
        _preload_cuda_libraries()
        try:
            from rapidocr_onnxruntime import RapidOCR
            if use_gpu:
                self.mode = "rapidocr_gpu"
                self.engine = RapidOCR(
                    det_use_cuda=True, 
                    cls_use_cuda=True, 
                    rec_use_cuda=True,
                    det_limit_side_len=960,
                    det_limit_type='max',
                    det_box_thresh=0.6,
                    det_unclip_ratio=1.5
                )
                logger.info("RapidOCR GPU engine initialized successfully with CUDA 12.")
            else:
                self.mode = "rapidocr_cpu"
                self.engine = RapidOCR(
                    det_use_cuda=False, 
                    cls_use_cuda=False, 
                    rec_use_cuda=False,
                    det_limit_side_len=960,
                    det_limit_type='max',
                    det_box_thresh=0.6,
                    det_unclip_ratio=1.5
                )
                logger.info("RapidOCR CPU engine initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize OCR engine: {e}")
            raise e

    def ocr(self, img, det: bool = True, rec: bool = True, cls: bool = False):
        """
        Run OCR on image and return standard PaddleOCR output format:
        [
            [
                [ [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], (text, confidence) ],
                ...
            ]
        ]
        """
        if isinstance(img, Path):
            img = str(img)

        # 1. If using native PaddleOCR (CPU mode)
        if self.mode == "paddleocr_cpu":
            return self.engine.ocr(img, det=det, rec=rec, cls=cls)

        # 2. If using RapidOCR (GPU / fallback)
        try:
            results, elapse = self.engine(img)
        except Exception as e:
            logger.error(f"OCR inference error: {e}")
            return [[]]

        if not results:
            return [[]]

        paddle_formatted = []
        for item in results:
            bbox = item[0]
            text = str(item[1])
            conf = float(item[2])

            if isinstance(bbox, np.ndarray):
                bbox = bbox.tolist()
            
            paddle_formatted.append([bbox, (text, conf)])

        return [paddle_formatted]
