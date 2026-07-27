import re
import logging
from .validator import clean_mrz_line, compute_mrz_check_digit, is_valid_fin, is_valid_id_number

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

        # Filter lines that look like MRZ lines (long enough)
        mrz_candidates = []
        for line_txt, conf in merged_lines:
            if len(line_txt) >= MRZ_LINE_MIN_LEN:
                mrz_candidates.append((line_txt, conf))

        if not mrz_candidates:
            logger.warning("No MRZ line candidates found")
            return MRZResult(
                fin=None, id_number=None, confidence=0.0, line1="", line2="", line3="",
                checksum_valid=False, method="not_found"
            )

        # Filter out VIZ noise lines (dates, addresses) that OCR picks up
        # alongside actual MRZ lines. Only keep structurally valid MRZ lines.
        validated = [(txt, conf) for txt, conf in mrz_candidates if self._is_mrz_line(txt)]
        if validated:
            dropped = len(mrz_candidates) - len(validated)
            if dropped > 0:
                logger.info(f"Filtered out {dropped} non-MRZ noise line(s)")
            mrz_candidates = validated
        else:
            logger.info("MRZ structural filter rejected all lines; using unfiltered candidates")

        # Run both extractors (Try-Both-and-Validate strategy)
        result_td1 = self._extract_td1(mrz_candidates, is_cropped)
        result_td2 = self._extract_td2(mrz_candidates, is_cropped)

        score_td1 = self._score_result(result_td1)
        score_td2 = self._score_result(result_td2)
        card_type_hint = self._classify_card_type(mrz_candidates)

        if score_td2 > score_td1:
            winner = result_td2
            winner_name = "TD2"
        elif score_td1 > score_td2:
            winner = result_td1
            winner_name = "TD1"
        else:
            # Score tie: pattern-based classifier breaks the tie
            winner = result_td2 if card_type_hint == "TD2" else result_td1
            winner_name = card_type_hint

        logger.info(f"Try-both scores: TD1={score_td1}, TD2={score_td2}, classifier_hint={card_type_hint} -> selected={winner_name}")
        return winner

    @staticmethod
    def _score_result(result) -> int:
        """
        Score an extraction result (MRZResult) using structural validation signals.
        """
        if not result or result.method == "not_found":
            return -100

        score = 0

        # 1. Checksum valid: +10 points (strongest signal)
        if result.checksum_valid:
            score += 10
        elif (result.line2 and len(result.line2) >= 10 and result.line2[9].isdigit()) or \
             (result.line1 and len(result.line1) >= 15 and result.line1[14].isdigit()):
            # Active checksum failure (digit present in check slot but calculation failed)
            score -= 3

        # 2. Valid FIN format: +5 points
        if result.fin and is_valid_fin(result.fin):
            score += 5

        # 3. Valid ID number format: +5 points
        if result.id_number and is_valid_id_number(result.id_number):
            score += 5

        # 4. Country code check: +3 points if 'AZE' found in appropriate slot
        if result.is_old_card:
            if len(result.line2) >= 13 and result.line2[10:13] == "AZE":
                score += 3
        else:
            if len(result.line1) >= 5 and result.line1[2:5] == "AZE":
                score += 3

        return score

    @staticmethod
    def _is_mrz_line(line_text: str) -> bool:
        """
        Check if a candidate text line looks like a genuine MRZ line.
        Filters out VIZ noise (dates, addresses, labels) that OCR picks up
        alongside actual MRZ lines on old ID cards.

        Valid MRZ line patterns:
        - TD1/TD2 Line 1: Starts with document type [IPAC], subtype/filler, country code
        - TD2 Line 2: Starts with doc number area (digits), country code at pos 10-12
        - TD1 Line 2: Starts with 6-digit DOB (YYMMDD) + check digit
        - TD1 Line 3: All uppercase letters and '<' fillers (name field)
        """
        if len(line_text) < 20:
            return False

        # MRZ Line 1 (TD1/TD2): starts with doc type indicator + subtype/filler + 3-letter country
        # e.g., "I<AZE...", "IAAZE...", "P<AZE..."
        if re.match(r'^[IPAC][<A-Z][A-Z]{3}', line_text):
            return True

        # MRZ Line 2 (TD2): doc number area (9 chars with at least some digits),
        # then check digit, then 3-letter country code at positions 10-12
        # e.g., "09163467<6AZE..."
        if len(line_text) >= 15:
            doc_area = line_text[0:9]
            has_digits_in_doc = any(c.isdigit() for c in doc_area)
            if has_digits_in_doc:
                country_area = line_text[10:13]
                if re.match(r'^[A-Z]{3}$', country_area):
                    return True

        # MRZ Line 2 (TD1): starts with DOB (6 digits YYMMDD) + check digit
        # e.g., "8303109M2901271AZE..."
        if re.match(r'^\d{7}', line_text):
            return True

        # MRZ Line 3 (TD1): name field — all uppercase letters and '<' fillers, no digits
        # Must contain '<<' double-filler separator (SURNAME<<GIVENNAMES)
        # Must end with trailing '<' padding (MRZ names are padded to 30 chars)
        # e.g., "HASANOV<<RAMIL<<<<<<<<<<<<<<<<<"
        if re.match(r'^[A-Z<]{20,}$', line_text) and '<<' in line_text:
            # Real MRZ name lines are padded to 30/36 chars and end with '<'
            # VIZ address lines (e.g., "AZRBAYCAN<BRDRAY<<BAYRAMLI") end with text, not padding
            if line_text.endswith('<<<'):
                return True

        return False

    @staticmethod
    def _detect_direct_pattern(mrz_candidates: list[tuple[str, float]]) -> str | None:
        """
        Check candidate lines for definitive direct ICAO document headers.
        Returns 'TD1' for 'IAAZE' (new cards), 'TD2' for 'I<AZE' (old cards), or None.
        """
        for line_text, _ in mrz_candidates:
            if re.match(r'^I[A-Z]AZE', line_text) or re.match(r'^I[A-Z<]AZE(AA|AB|\d)', line_text):
                return "TD1"

            if line_text.startswith("I<AZE") or re.match(r'^I<[A-Z]{3}[A-Z<]+', line_text):
                after_aze = line_text[5:15] if len(line_text) >= 15 else line_text[5:]
                if not any(c.isdigit() for c in after_aze):
                    return "TD2"

        return None

    def _classify_card_type(self, mrz_candidates: list[tuple[str, float]]) -> str:
        """
        Classify MRZ card format directly via ICAO document type prefix matching:
        - New Biometric Card (TD1): Line 1 starts with 'IAAZE' or 'I[A-Z]AZE' followed by ID digits or AA/AB prefix.
        - Old Card (TD2): Line 1 starts with 'I<AZE' followed by surname text.
        """
        for line_text, _ in mrz_candidates:
            # Check for TD1 (New Card) Line 1 pattern: e.g., IAAZEAA3778866...
            if re.match(r'^I[A-Z]AZE', line_text) or re.match(r'^I[A-Z<]AZE(AA|AB|\d)', line_text):
                logger.info(f"Direct pattern match: TD1 Line 1 identified ('{line_text[:10]}')")
                return "TD1"

            # Check for TD2 (Old Card) Line 1 pattern: e.g., I<AZEBAYRAMOV...
            if line_text.startswith("I<AZE") or re.match(r'^I<[A-Z]{3}[A-Z<]+', line_text):
                after_aze = line_text[5:15] if len(line_text) >= 15 else line_text[5:]
                if not any(c.isdigit() for c in after_aze):
                    logger.info(f"Direct pattern match: TD2 Line 1 identified ('{line_text[:10]}')")
                    return "TD2"

        # Fallback to line count and length heuristics
        n = len(mrz_candidates)
        if n >= 3:
            last3 = mrz_candidates[-3:]
            avg_len = sum(len(l[0]) for l in last3) / 3.0
            if avg_len <= 33:
                return "TD1"

        if n == 2:
            avg_len = sum(len(l[0]) for l in mrz_candidates) / 2.0
            if avg_len >= 33:
                return "TD2"

        first_line = mrz_candidates[0][0] if mrz_candidates else ""
        after_pos5 = first_line[5:] if len(first_line) > 5 else ""
        has_digits = any(c.isdigit() for c in after_pos5[:9])

        if has_digits:
            return "TD1"
        return "TD2"

    def _extract_td1(self, mrz_candidates: list[tuple[str, float]], is_cropped: bool):
        """Extract FIN and ID number from TD1 (3-line, new biometric card) MRZ."""
        from . import MRZResult

        # Pick the last 3 candidates for TD1
        candidates = mrz_candidates[-3:] if len(mrz_candidates) >= 3 else mrz_candidates[:]
        while len(candidates) < 3:
            candidates.append(("", 0.0))

        line1_raw, l1_conf = candidates[0]
        line2_raw, l2_conf = candidates[1]
        line3_raw, l3_conf = candidates[2]

        logger.info(f"Reconstructed TD1 Line 1: '{line1_raw}'")
        logger.info(f"Reconstructed TD1 Line 2: '{line2_raw}'")
        logger.info(f"Reconstructed TD1 Line 3: '{line3_raw}'")

        def normalize_mrz_len(s: str) -> str:
            if len(s) < MRZ_LINE_EXPECTED_LEN:
                return s + "<" * (MRZ_LINE_EXPECTED_LEN - len(s))
            return s[:MRZ_LINE_EXPECTED_LEN]

        line1 = normalize_mrz_len(line1_raw)
        line2 = normalize_mrz_len(line2_raw)
        line3 = normalize_mrz_len(line3_raw)

        # Extract ID number from Line 1 (positions 5-13, index 5 to 14)
        doc_num_raw = line1[5:14].rstrip('<')
        id_number = doc_num_raw if doc_num_raw else None
        logger.info(f"Extracted TD1 ID Number: '{id_number}'")

        # Extract FIN from Line 1 optional data
        issuing_state = line1[2:5]
        if not issuing_state.isalnum():
            issuing_state = "AZE"

        fin_candidates = []
        for offset in [15, 14, 13, 16]:
            if len(line1) <= offset:
                continue
            
            raw_opt = line1[offset:29]
            opt_clean = raw_opt.rstrip('<')
            if not opt_clean:
                continue

            if opt_clean.startswith(issuing_state) and len(opt_clean) >= len(issuing_state) + 7:
                val = opt_clean[len(issuing_state) : len(issuing_state) + 7]
                if re.match(r'^[A-Z0-9]{7}$', val):
                    score = 10 + (2 if offset == 15 else 0)
                    fin_candidates.append((val, score, offset))

            if len(opt_clean) == 7 and re.match(r'^[A-Z0-9]{7}$', opt_clean):
                score = 8 + (2 if offset == 15 else 0)
                fin_candidates.append((opt_clean, score, offset))

            sub_matches = re.findall(r'[A-Z0-9]{7}', opt_clean)
            for m in sub_matches:
                if m != issuing_state:
                    fin_candidates.append((m, 5, offset))

            sub_matches_fallback = re.findall(r'[A-Z0-9]{5,8}', opt_clean)
            for m in sub_matches_fallback:
                if m != issuing_state:
                    fin_candidates.append((m, 3, offset))

        fin_candidate = None
        if fin_candidates:
            fin_candidates.sort(key=lambda x: x[0])
            fin_candidates.sort(key=lambda x: x[1], reverse=True)
            fin_candidate = fin_candidates[0][0]
            logger.info(f"Selected TD1 FIN candidate: '{fin_candidate}'")

        # Checksum validation for doc number
        checksum_valid = False
        if len(line1) >= 15:
            doc_num = line1[5:14]
            expected_cd_str = line1[14]
            if doc_num and expected_cd_str.isdigit():
                calculated_cd = compute_mrz_check_digit(doc_num)
                checksum_valid = (calculated_cd == int(expected_cd_str))

        avg_mrz_confidence = (l1_conf + l2_conf + l3_conf) / 3.0 if (l1_conf and l2_conf and l3_conf) else l1_conf
        mrz_method = "mrz_strip" if is_cropped else "full_image_fallback"
        if not fin_candidate:
            mrz_method = "not_found"

        return MRZResult(
            fin=fin_candidate,
            id_number=id_number,
            confidence=avg_mrz_confidence,
            line1=line1,
            line2=line2,
            line3=line3,
            checksum_valid=checksum_valid,
            method=mrz_method
        )

    def _extract_td2(self, mrz_candidates: list[tuple[str, float]], is_cropped: bool):
        """Extract FIN and ID number from TD2 (2-line, old card) MRZ."""
        from . import MRZResult

        TD2_EXPECTED_LEN = 36

        candidates = mrz_candidates[-2:] if len(mrz_candidates) >= 2 else mrz_candidates[:]
        while len(candidates) < 2:
            candidates.append(("", 0.0))

        line1_raw, l1_conf = candidates[0]
        line2_raw, l2_conf = candidates[1]

        def looks_like_td2_line2(txt: str) -> bool:
            if not txt:
                return False
            if re.match(r'^\d{5,}', txt):
                return True
            if len(txt) >= 13:
                doc_area = txt[:9]
                digit_cnt = sum(1 for ch in doc_area if ch.isdigit())
                if digit_cnt >= 4 and txt[10:13].isalpha():
                    return True
            return False

        if looks_like_td2_line2(line1_raw) and not looks_like_td2_line2(line2_raw):
            logger.info("Content-aware TD2: Swapping candidates (Line 1 has doc number line)")
            line1_raw, line2_raw = line2_raw, line1_raw
            l1_conf, l2_conf = l2_conf, l1_conf
        elif looks_like_td2_line2(line1_raw) and line2_raw == "":
            logger.info("Content-aware TD2: Single candidate detected is doc number line -> assigned to Line 2")
            line2_raw = line1_raw
            l2_conf = l1_conf
            line1_raw = ""
            l1_conf = 0.0

        logger.info(f"Reconstructed TD2 Line 1: '{line1_raw}'")
        logger.info(f"Reconstructed TD2 Line 2: '{line2_raw}'")

        def normalize_td2(s: str) -> str:
            if len(s) < TD2_EXPECTED_LEN:
                return s + "<" * (TD2_EXPECTED_LEN - len(s))
            return s[:TD2_EXPECTED_LEN]

        line1 = normalize_td2(line1_raw)
        line2 = normalize_td2(line2_raw)

        # Extract Doc Number and Nationality from Line 2
        # Doc Number: positions 0-8 (9 chars)
        doc_num_raw = line2[0:9].rstrip('<')
        nationality = line2[10:13]
        if not nationality.isalpha():
            nationality = "AZE"

        id_number = (nationality + doc_num_raw) if doc_num_raw else None
        logger.info(f"Extracted TD2 ID Number: '{id_number}' (doc: '{doc_num_raw}', nat: '{nationality}')")

        # Extract FIN from Line 2 Optional Data (positions 28-34, 7 chars)
        fin_candidate = None
        
        # Try exact 28-35 chunk
        raw_opt = line2[28:35].rstrip('<')
        if len(raw_opt) == 7 and re.match(r'^[A-Z0-9]{7}$', raw_opt):
            fin_candidate = raw_opt
        elif len(raw_opt) > 0:
            m = re.search(r'[A-Z0-9]{7}', raw_opt)
            if m:
                fin_candidate = m.group()

        # Fallback: scan positions 27 to 36 for any 7-char alphanumeric string
        if not fin_candidate and len(line2) >= 35:
            opt_window = line2[27:36].rstrip('<')
            m = re.search(r'[A-Z0-9]{7}', opt_window)
            if m:
                fin_candidate = m.group()

        # Checksum validation for doc number (position 9 is check digit for positions 0-8)
        checksum_valid = False
        if len(line2) >= 10:
            doc_num_field = line2[0:9]
            expected_cd_str = line2[9]
            if doc_num_field and expected_cd_str.isdigit():
                calculated_cd = compute_mrz_check_digit(doc_num_field)
                checksum_valid = (calculated_cd == int(expected_cd_str))

        avg_mrz_confidence = (l1_conf + l2_conf) / 2.0 if (l1_conf and l2_conf) else max(l1_conf, l2_conf)
        mrz_method = "td2_extract" if is_cropped else "td2_full_image_fallback"
        if not fin_candidate and not id_number:
            mrz_method = "not_found"

        return MRZResult(
            fin=fin_candidate,
            id_number=id_number,
            confidence=avg_mrz_confidence,
            line1=line1,
            line2=line2,
            line3="",
            checksum_valid=checksum_valid,
            method=mrz_method
        )
