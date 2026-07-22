import re
import logging
from .validator import clean_ocr_text, is_valid_fin

logger = logging.getLogger(__name__)

# Constants
VIZ_LABEL_PATTERNS = [
    "fin", 
    "personalno", 
    "ferdino", 
    "personalnumber", 
    "fardino", 
    "ferdi", 
    "ferdiidentifikasiyanomresipersonalno",
    "ferdiidentifikasiyanomresi",
    "personal"
]
FIN_REGEX_CANDIDATE = re.compile(r'^[A-Z0-9]{5,8}$')
FIN_REGEX_BLIND = re.compile(r'^[A-Z0-9]{7}$')

def get_edit_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def normalize_text_for_label(text: str) -> str:
    """Normalize text for label matching (lower, no spaces/punctuation, ə -> e, ö -> o, ı -> i)."""
    text = text.lower().strip()
    text = text.replace("ə", "e")
    text = text.replace("ö", "o")
    text = text.replace("ı", "i")
    text = re.sub(r'[^a-z0-9]', '', text)
    return text

class VIZExtractor:
    """Extractor for Azerbaijani ID Card FIN code from the VIZ (front) side."""

    def __init__(self, ocr_engine):
        """
        Initialize VIZExtractor with a pre-configured PaddleOCR engine instance.
        """
        self.ocr = ocr_engine

    def extract(self, img):
        """
        Extract the FIN code from VIZ image.
        Returns VIZResult.
        """
        from . import VIZResult
        
        try:
            # Run OCR on the image
            ocr_results = self.ocr.ocr(img, det=True, rec=True, cls=False)
        except Exception as e:
            logger.error(f"PaddleOCR invocation failed in VIZExtractor: {e}")
            return VIZResult(fin=None, confidence=0.0, bbox=None, method="not_found")

        if not ocr_results or not ocr_results[0]:
            logger.warning("No text detected by PaddleOCR on VIZ side")
            return VIZResult(fin=None, confidence=0.0, bbox=None, method="not_found")

        # PaddleOCR returns a list of results (one per image). We process the first image.
        blocks = []
        for line in ocr_results[0]:
            bbox = line[0]  # 4 points
            text_str, conf = line[1]
            text_str = clean_ocr_text(text_str)
            blocks.append({
                "text": text_str,
                "bbox": bbox,
                "confidence": conf,
                "xmin": min(p[0] for p in bbox),
                "xmax": max(p[0] for p in bbox),
                "ymin": min(p[1] for p in bbox),
                "ymax": max(p[1] for p in bbox),
                "width": max(p[0] for p in bbox) - min(p[0] for p in bbox),
                "height": max(p[1] for p in bbox) - min(p[1] for p in bbox),
                "cx": sum(p[0] for p in bbox) / 4.0,
                "cy": sum(p[1] for p in bbox) / 4.0
            })

        # Step 1: Search for label
        label_block = None
        min_dist = 9999
        
        # Exact/prefix matching first
        for block in blocks:
            norm_txt = normalize_text_for_label(block["text"])
            if any(p in norm_txt for p in VIZ_LABEL_PATTERNS) or any(norm_txt in p for p in VIZ_LABEL_PATTERNS):
                label_block = block
                break
                
        # Fuzzy matching if exact label not found
        if not label_block:
            for block in blocks:
                norm_txt = normalize_text_for_label(block["text"])
                for pattern in VIZ_LABEL_PATTERNS:
                    dist = get_edit_distance(norm_txt, pattern)
                    if dist <= 1:
                        if dist < min_dist:
                            min_dist = dist
                            label_block = block

        # Step 2: Extract value based on proximity if label found
        if label_block:
            logger.info(f"Detected VIZ label: '{label_block['text']}' at {label_block['bbox']}")
            label_height = label_block["height"]
            
            candidates = []
            for block in blocks:
                # Do not evaluate the label itself
                if block is label_block:
                    continue
                
                # Check candidate matches FIN format
                text = block["text"]
                if not FIN_REGEX_CANDIDATE.match(text):
                    continue
                    
                # Proximity rule 1: Same horizontal line to the right
                # - Vertically within 1.5 * label_height
                # - Horizontally to the right (candidate_xmin >= label_xmax) and within 250px
                y_diff = abs(block["cy"] - label_block["cy"])
                x_diff = block["xmin"] - label_block["xmax"]
                
                is_right = y_diff <= 1.5 * label_height and 0 <= x_diff <= 250
                
                # Proximity rule 2: Directly below
                # - Vertically below (candidate_ymin >= label_ymax) and within 2.0 * label_height
                # - Horizontally overlapping or close
                y_below = block["ymin"] - label_block["ymax"]
                x_overlap_width = min(block["xmax"], label_block["xmax"]) - max(block["xmin"], label_block["xmin"])
                has_overlap = x_overlap_width > 0 or abs(block["cx"] - label_block["cx"]) <= label_block["width"]
                
                is_below = 0 <= y_below <= 2.0 * label_height and has_overlap
                
                if is_right or is_below:
                    # Calculate proximity score (closer is better)
                    dist = ((block["cx"] - label_block["cx"]) ** 2 + (block["cy"] - label_block["cy"]) ** 2) ** 0.5
                    candidates.append((block, dist))
                    
            if candidates:
                # Sort candidates by proximity distance
                candidates.sort(key=lambda x: x[1])
                best_candidate, _ = candidates[0]
                logger.info(f"Found VIZ FIN candidate by proximity: '{best_candidate['text']}' (conf: {best_candidate['confidence']})")
                return VIZResult(
                    fin=best_candidate["text"],
                    confidence=best_candidate["confidence"],
                    bbox=best_candidate["bbox"],
                    method="label_proximity"
                )

        # Step 3: Blind regex scan fallback
        logger.info("VIZ label proximity search failed, performing blind regex scan")
        blind_candidates = []
        for block in blocks:
            text = block["text"]
            # Exclude strings that look like document numbers or other long/short texts
            # Document numbers are typically 8-9 chars or start differently.
            # We want exact 7 characters for blind scan.
            if FIN_REGEX_BLIND.match(text) and block["confidence"] >= 0.70:
                blind_candidates.append(block)

        if blind_candidates:
            # Sort by confidence descending
            blind_candidates.sort(key=lambda x: x["confidence"], reverse=True)
            best_blind = blind_candidates[0]
            logger.info(f"Found VIZ FIN candidate by blind regex: '{best_blind['text']}' (conf: {best_blind['confidence']})")
            return VIZResult(
                fin=best_blind["text"],
                confidence=best_blind["confidence"],
                bbox=best_blind["bbox"],
                method="blind_regex"
            )

        logger.warning("No FIN code found on VIZ side")
        return VIZResult(fin=None, confidence=0.0, bbox=None, method="not_found")
