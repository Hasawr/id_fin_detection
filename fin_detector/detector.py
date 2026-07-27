import os
import glob
import site
import ctypes
import logging
from pathlib import Path

# Pre-load NVIDIA CUDA/cuDNN libraries from virtualenv into global symbol table
try:
    site_pkg = site.getsitepackages()[0]
    nvidia_dirs = glob.glob(os.path.join(site_pkg, 'nvidia', '*', 'lib'))
    if nvidia_dirs:
        os.environ['LD_LIBRARY_PATH'] = ':'.join(nvidia_dirs) + ':' + os.environ.get('LD_LIBRARY_PATH', '')
        for d in nvidia_dirs:
            for so_file in glob.glob(os.path.join(d, '*.so*')):
                try:
                    ctypes.CDLL(so_file, mode=ctypes.RTLD_GLOBAL)
                except Exception:
                    pass
except Exception:
    pass

from .ocr_engine import OCREngineAdapter

from .preprocessor import ImagePreprocessor
from .viz_extractor import VIZExtractor
from .mrz_extractor import MRZExtractor
from . import VIZResult, MRZResult, FINDetectionOutput
from utils.image_utils import draw_viz_debug, draw_mrz_debug, save_debug_image

logger = logging.getLogger(__name__)

class FINDetector:
    """Orchestrator class to run FIN code detection on VIZ and/or MRZ sides of ID cards."""

    def __init__(self, use_gpu: bool = True, debug: bool = False):
        """
        Initialize OCR engine and extractor components on GPU/CPU.
        """
        logger.info(f"Initializing FINDetector (use_gpu={use_gpu}, debug={debug})...")
        
        # Initialize OCR engine strictly on requested device
        self.ocr_engine = OCREngineAdapter(use_gpu=use_gpu)
        
        self.preprocessor = ImagePreprocessor()
        self.viz_extractor = VIZExtractor(self.ocr_engine)
        self.mrz_extractor = MRZExtractor(self.ocr_engine)
        self.debug = debug
        self.use_gpu = use_gpu

    def detect_from_viz(self, image_path: str | Path) -> FINDetectionOutput:
        """
        Run FIN detection on the VIZ (front) side of the ID card.
        """
        image_path = Path(image_path)
        logger.info(f"Processing VIZ side from: {image_path.name}")
        notes = []
        
        try:
            # 1. Load image
            img = self.preprocessor.load(image_path)
            
            # 2. Deskew / Rectify card boundaries
            rectified = self.preprocessor.detect_card_roi(img)
            if rectified is img:
                notes.append("Card ROI boundary not detected; using original image.")
            else:
                notes.append("Card ROI successfully detected and rectified.")
                
            # 3. Enhance for VIZ (grayscale/contrast/sharpen)
            enhanced = self.preprocessor.enhance_for_viz(rectified)
            
            # 4. Extract
            viz_result = self.viz_extractor.extract(enhanced)
            
            # 5. Debug output
            if self.debug and viz_result.fin:
                annotated = draw_viz_debug(rectified, viz_result.bbox, viz_result.bbox, viz_result.fin)
                save_debug_image(annotated, f"debug_viz_{image_path.name}")
                notes.append(f"Saved debug image to debug_output/debug_viz_{image_path.name}")
                
            return FINDetectionOutput(
                viz_fin=viz_result.fin,
                mrz_fin=None,
                viz_confidence=viz_result.confidence,
                mrz_confidence=0.0,
                viz_result=viz_result,
                mrz_result=None,
                notes=notes
            )
            
        except Exception as e:
            logger.error(f"Error processing VIZ image {image_path.name}: {e}", exc_info=True)
            return FINDetectionOutput(
                viz_fin=None,
                mrz_fin=None,
                viz_confidence=0.0,
                mrz_confidence=0.0,
                notes=[f"Failed to process VIZ image: {str(e)}"]
            )

    def detect_from_mrz(self, image_path: str | Path) -> FINDetectionOutput:
        """
        Run FIN detection on the MRZ (back) side of the ID card.
        """
        image_path = Path(image_path)
        logger.info(f"Processing MRZ side from: {image_path.name}")
        notes = []
        
        try:
            # 1. Load image
            img = self.preprocessor.load(image_path)
            
            # 2. Deskew / Rectify card boundaries
            rectified = self.preprocessor.detect_card_roi(img)
            if rectified is img:
                notes.append("Card ROI boundary not detected; using original image.")
            else:
                notes.append("Card ROI successfully detected and rectified.")
                
            # 3. Try cropped bottom 25% MRZ first
            enhanced_mrz = self.preprocessor.enhance_for_mrz(rectified)
            mrz_result = self.mrz_extractor.extract(enhanced_mrz, is_cropped=True)
            
            # 4. Fallback sequence if cropped 25% failed to extract valid FIN
            if not mrz_result.fin or len(mrz_result.fin) != 7:
                logger.info("FIN not found in 25% MRZ crop; trying wider bottom 45% crop...")
                enhanced_wide = self.preprocessor.enhance_for_mrz_wide(rectified)
                mrz_result = self.mrz_extractor.extract(enhanced_wide, is_cropped=True)
                notes.append("Tried wider 45% MRZ crop fallback.")

            if not mrz_result.fin or len(mrz_result.fin) != 7:
                logger.info("FIN not found in cropped MRZ bands; trying fallback on full rectified image")
                mrz_result = self.mrz_extractor.extract(rectified, is_cropped=False)
                notes.append("FIN extraction fell back to full image scan.")
            else:
                notes.append("FIN successfully extracted from cropped MRZ band.")
                
            if mrz_result.is_old_card:
                notes.append(f"Card format: {mrz_result.card_format}")
                
            # 5. Debug output
            if self.debug:
                annotated = draw_mrz_debug(
                    rectified, 
                    mrz_result.line1, 
                    mrz_result.line2, 
                    mrz_result.line3, 
                    mrz_result.fin or "NOT_FOUND",
                    id_number=mrz_result.id_number or "NOT_FOUND",
                    card_type=mrz_result.card_format
                )
                save_debug_image(annotated, f"debug_mrz_{image_path.name}")
                notes.append(f"Saved debug image to debug_output/debug_mrz_{image_path.name}")
                
            return FINDetectionOutput(
                viz_fin=None,
                mrz_fin=mrz_result.fin,
                viz_confidence=0.0,
                mrz_confidence=mrz_result.confidence,
                mrz_id_number=mrz_result.id_number,
                viz_result=None,
                mrz_result=mrz_result,
                notes=notes
            )
            
        except Exception as e:
            logger.error(f"Error processing MRZ image {image_path.name}: {e}", exc_info=True)
            return FINDetectionOutput(
                viz_fin=None,
                mrz_fin=None,
                viz_confidence=0.0,
                mrz_confidence=0.0,
                mrz_id_number=None,
                notes=[f"Failed to process MRZ image: {str(e)}"]
            )

    def detect_from_both(self, viz_image_path: str | Path, mrz_image_path: str | Path) -> FINDetectionOutput:
        """
        Run FIN detection on both sides and return the results side-by-side.
        """
        viz_out = self.detect_from_viz(viz_image_path)
        mrz_out = self.detect_from_mrz(mrz_image_path)
        
        combined_notes = []
        combined_notes.extend([f"[VIZ] {n}" for n in viz_out.notes])
        combined_notes.extend([f"[MRZ] {n}" for n in mrz_out.notes])
        
        return FINDetectionOutput(
            viz_fin=viz_out.viz_fin,
            mrz_fin=mrz_out.mrz_fin,
            viz_confidence=viz_out.viz_confidence,
            mrz_confidence=mrz_out.mrz_confidence,
            mrz_id_number=mrz_out.mrz_id_number,
            viz_result=viz_out.viz_result,
            mrz_result=mrz_out.mrz_result,
            notes=combined_notes
        )
