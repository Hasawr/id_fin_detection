import re
import logging
from .validator import clean_mrz_line, compute_mrz_check_digit, is_valid_fin

logger = logging.getLogger(__name__)

# Constants
MRZ_LINE_MIN_LEN = 10
MRZ_LINE_EXPECTED_LEN = 30

class MRZExtractor:
    """Extractor for Azerbaijani ID Card FIN code from the MRZ (back) side."""

    def __init__(self, ocr_engine):
        """
        Initialize MRZExtractor with a pre-configured PaddleOCR engine instance.
        """
        self.ocr = ocr_engine

    def extract(self, img, is_cropped=False):
        """
        Extract the FIN code from the MRZ image.
        Supports both full back-side image or pre-cropped MRZ region.
        Returns MRZResult.
        """
        from . import MRZResult

        try:
            # Run OCR on the image
            ocr_results = self.ocr.ocr(img, det=True, rec=True, cls=False)
        except Exception as e:
            logger.error(f"PaddleOCR invocation failed in MRZExtractor: {e}")
            return MRZResult(
                fin=None, confidence=0.0, line1="", line2="", line3="",
                checksum_valid=False, method="not_found"
            )

        if not ocr_results or not ocr_results[0]:
            logger.warning("No text detected by PaddleOCR on MRZ side")
            return MRZResult(
                fin=None, confidence=0.0, line1="", line2="", line3="",
                checksum_valid=False, method="not_found"
            )

        # Parse detected text blocks
        blocks = []
        for line in ocr_results[0]:
            bbox = line[0]
            text_str, conf = line[1]
            text_cleaned = clean_mrz_line(text_str)
            if not text_cleaned:
                continue
            
            blocks.append({
                "text": text_cleaned,
                "bbox": bbox,
                "confidence": conf,
                "cy": sum(p[1] for p in bbox) / 4.0,
                "xmin": min(p[0] for p in bbox)
            })

        if not blocks:
            logger.warning("No valid text blocks found after cleaning MRZ")
            return MRZResult(
                fin=None, confidence=0.0, line1="", line2="", line3="",
                checksum_valid=False, method="not_found"
            )

        # Reconstruct MRZ lines
        # Group blocks that are on the same horizontal line (within vertical threshold)
        # We sort blocks by Y coordinate first to group them
        blocks.sort(key=lambda x: x["cy"])
        
        mrz_lines_grouped = []
        current_line = []
        
        # Calculate average height for Y distance thresholding
        avg_height = sum(
            (max(b["bbox"][i][1] for i in range(4)) - min(b["bbox"][i][1] for i in range(4))) 
            for b in blocks
        ) / len(blocks)
        
        y_threshold = avg_height * 0.7
        
        for b in blocks:
            if not current_line:
                current_line.append(b)
            else:
                # If current block center Y is close to the previous block center Y, group them
                if abs(b["cy"] - current_line[-1]["cy"]) < y_threshold:
                    current_line.append(b)
                else:
                    mrz_lines_grouped.append(current_line)
                    current_line = [b]
        if current_line:
            mrz_lines_grouped.append(current_line)

        # For each line group, sort left-to-right and merge
        merged_lines = []
        for group in mrz_lines_grouped:
            group.sort(key=lambda x: x["xmin"])
            line_text = "".join(b["text"] for b in group)
            avg_conf = sum(b["confidence"] for b in group) / len(group)
            merged_lines.append((line_text, avg_conf))

        # Filter lines that look like MRZ lines (long, contain '<')
        mrz_candidates = []
        for line_txt, conf in merged_lines:
            # Standard TD1 lines are 30 chars. Allow slightly shorter/longer due to OCR errors.
            if len(line_txt) >= MRZ_LINE_MIN_LEN:
                mrz_candidates.append((line_txt, conf))

        # We need 3 lines for TD1. If we have more or less, try to pick the best 3.
        # For TD1, the MRZ is at the bottom of the card.
        # We sort by Y position (already done implicitly by grouping).
        # We take the last 3 candidates if possible, as the MRZ is at the very bottom.
        if len(mrz_candidates) >= 3:
            mrz_candidates = mrz_candidates[-3:]
        
        # If we couldn't find 3 lines, let's pad them
        while len(mrz_candidates) < 3:
            mrz_candidates.append(("", 0.0))

        line1_raw, l1_conf = mrz_candidates[0]
        line2_raw, l2_conf = mrz_candidates[1]
        line3_raw, l3_conf = mrz_candidates[2]

        logger.info(f"Reconstructed MRZ Line 1: '{line1_raw}'")
        logger.info(f"Reconstructed MRZ Line 2: '{line2_raw}'")
        logger.info(f"Reconstructed MRZ Line 3: '{line3_raw}'")

        # Normalize line lengths to 30 characters
        def normalize_mrz_len(s: str) -> str:
            if len(s) < MRZ_LINE_EXPECTED_LEN:
                return s + "<" * (MRZ_LINE_EXPECTED_LEN - len(s))
            return s[:MRZ_LINE_EXPECTED_LEN]

        line1 = normalize_mrz_len(line1_raw)
        line2 = normalize_mrz_len(line2_raw)
        line3 = normalize_mrz_len(line3_raw)

        # Extract FIN from Line 1 using candidate scoring across multiple common offsets:
        # Standard TD1 optional data starts at index 15 (position 16).
        # Mock/non-standard layout from prompt starts at index 14.
        # We also try index 13 and 16 to handle OCR shifts.
        issuing_state = line1[2:5]
        if not issuing_state.isalnum():
            issuing_state = "AZE"

        candidates = []
        
        for offset in [15, 14, 13, 16]:
            if len(line1) <= offset:
                continue
            
            raw_opt = line1[offset:29]
            opt_clean = raw_opt.rstrip('<')
            if not opt_clean:
                continue
                
            # Case 1: starts with issuing state (e.g. AZE) -> skip it to get the 7-char FIN
            if opt_clean.startswith(issuing_state) and len(opt_clean) >= len(issuing_state) + 7:
                val = opt_clean[len(issuing_state) : len(issuing_state) + 7]
                if re.match(r'^[A-Z0-9]{7}$', val):
                    # Score is highest since country code is matched and stripped.
                    # Prefer index 15 (standard) over others if scores are identical.
                    score = 10 + (2 if offset == 15 else 0)
                    candidates.append((val, score, offset))
                    
            # Case 2: exactly 7 chars, no country code prefix
            if len(opt_clean) == 7 and re.match(r'^[A-Z0-9]{7}$', opt_clean):
                score = 8 + (2 if offset == 15 else 0)
                candidates.append((opt_clean, score, offset))
                
            # Case 3: any 7-character alphanumeric substring
            sub_matches = re.findall(r'[A-Z0-9]{7}', opt_clean)
            for m in sub_matches:
                if m != issuing_state:
                    candidates.append((m, 5, offset))
                    
            # Case 4: any 5-8 character alphanumeric fallback
            sub_matches_fallback = re.findall(r'[A-Z0-9]{5,8}', opt_clean)
            for m in sub_matches_fallback:
                if m != issuing_state:
                    candidates.append((m, 3, offset))

        fin_candidate = None
        best_offset = 15
        
        if candidates:
            # Sort candidates by score (descending)
            candidates.sort(key=lambda x: x[0])  # secondary sort: alphabetically/lexically
            candidates.sort(key=lambda x: x[1], reverse=True)  # primary sort: score descending
            fin_candidate, best_score, best_offset = candidates[0]
            logger.info(f"Selected best FIN candidate '{fin_candidate}' with score {best_score} from offset {best_offset}")
        else:
            logger.warning("No FIN candidates found in Line 1 optional data")

        # Perform document number checksum validation (positions 6-14, index 5-13; checksum is index 14)
        checksum_valid = False
        if len(line1) >= 15:
            doc_num = line1[5:14]
            expected_cd_str = line1[14]
            if doc_num and expected_cd_str.isdigit():
                calculated_cd = compute_mrz_check_digit(doc_num)
                checksum_valid = (calculated_cd == int(expected_cd_str))
                logger.info(f"Document No: '{doc_num}', Expected CD: '{expected_cd_str}', Calculated CD: '{calculated_cd}', Valid: {checksum_valid}")

        avg_mrz_confidence = (l1_conf + l2_conf + l3_conf) / 3.0 if (l1_conf and l2_conf and l3_conf) else l1_conf

        # Override method based on where we found it
        mrz_method = "mrz_strip" if fin_candidate else "not_found"
        if is_cropped and mrz_method != "not_found":
            mrz_method = "mrz_strip"
        elif not is_cropped and mrz_method != "not_found":
            mrz_method = "full_image_fallback"

        return MRZResult(
            fin=fin_candidate,
            confidence=avg_mrz_confidence,
            line1=line1,
            line2=line2,
            line3=line3,
            checksum_valid=checksum_valid,
            method=mrz_method
        )
