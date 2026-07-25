from dataclasses import dataclass
import logging
import re

from . import MRZResult
from .layout import (
    TD1_FIN,
    TD1_ISSUING_STATE,
    TD1_LINE_LENGTH,
    TD1_OPTIONAL_END,
    TD1_SERIAL,
    TD1_SERIAL_CHECK,
    TD1_SERIAL_SEARCH_STARTS,
    TD2_COMPOSITE_CHECK,
    TD2_FIN,
    TD2_FIN_AFTER_NATIONALITY,
    TD2_LINE_LENGTH,
    TD2_NATIONALITY,
    TD2_NATIONALITY_ALLOWED,
    TD2_SERIAL,
    TD2_SERIAL_CHECK,
)
from .validator import (
    clean_mrz_line,
    compute_mrz_check_digit,
    correct_ocr_digits,
    is_valid_fin,
    resolve_fin_o0_ambiguity,
    is_valid_new_card_serial,
    is_valid_old_card_serial,
)


logger = logging.getLogger(__name__)
MRZ_LINE_MIN_LENGTH = 10


@dataclass(frozen=True)
class MRZCandidate:
    format_name: str
    lines: tuple[tuple[str, float], ...]
    score: float


class MRZExtractor:
    def __init__(self, ocr_engine):
        self.ocr = ocr_engine

    def extract(
        self,
        image,
        *,
        attempt: str,
    ) -> MRZResult:
        try:
            ocr_results = self.ocr.ocr(image, det=True, rec=True, cls=False)
        except Exception:
            logger.exception("PaddleOCR invocation failed in MRZExtractor")
            return self._not_found()

        if not ocr_results or not ocr_results[0]:
            return self._not_found()

        blocks = []
        for line in ocr_results[0]:
            bbox = line[0]
            text, confidence = line[1]
            cleaned = clean_mrz_line(text)
            if cleaned:
                blocks.append(
                    {
                        "text": cleaned,
                        "bbox": bbox,
                        "confidence": confidence,
                        "cy": sum(point[1] for point in bbox) / 4.0,
                        "xmin": min(point[0] for point in bbox),
                    }
                )
        if not blocks:
            return self._not_found()

        merged_lines = self._merge_blocks(blocks)
        merged_lines = self._prefer_mrz_like_lines(merged_lines)
        return self._select_best_result(merged_lines, attempt)

    def extract_recognition_lines(
        self,
        line_images: list,
        *,
        attempt: str,
    ) -> MRZResult:
        """Recognize pre-cropped MRZ lines when text detection found nothing."""
        if len(line_images) != 2:
            return self._not_found()
        try:
            raw_results = self.ocr.ocr(
                line_images,
                det=False,
                rec=True,
                cls=False,
            )
        except Exception:
            logger.exception("PaddleOCR line recognition recovery failed")
            return self._not_found()

        recognition_results = (
            raw_results[0]
            if raw_results
            and len(raw_results) == 1
            and isinstance(raw_results[0], list)
            else raw_results
        )
        if not recognition_results:
            return self._not_found()

        lines: list[tuple[str, float]] = []
        for result in recognition_results:
            if not isinstance(result, (list, tuple)) or len(result) != 2:
                continue
            text, confidence = result
            cleaned = clean_mrz_line(str(text))
            if cleaned:
                lines.append((cleaned, float(confidence)))
        if len(lines) != 2:
            return self._diagnostic_result(lines)
        return self._select_best_result(lines, attempt)

    @classmethod
    def _mrz_line_score(cls, text: str) -> float:
        """Score how MRZ-like a cleaned OCR line is vs normal card prose."""
        if not text:
            return -10.0
        filler_ratio = text.count("<") / max(len(text), 1)
        has_header = cls._has_document_header(text)
        has_aze = "AZE" in text
        has_name_sep = "<<" in text
        digit_ratio = sum(char.isdigit() for char in text) / max(len(text), 1)
        # Long prose with almost no fillers is usually top-of-card instructions.
        prose_penalty = (
            -8.0
            if len(text) > 40 and filler_ratio < 0.08 and not has_aze
            else 0.0
        )
        return (
            filler_ratio * 12
            + (8 if has_header else 0)
            + (4 if has_aze else 0)
            + (3 if has_name_sep else 0)
            + digit_ratio * 4
            + min(len(text), 40) / 8
            + prose_penalty
        )

    @classmethod
    def _prefer_mrz_like_lines(
        cls,
        merged_lines: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        if len(merged_lines) <= 3:
            return merged_lines
        scored = [
            (index, line, cls._mrz_line_score(line[0]))
            for index, line in enumerate(merged_lines)
        ]
        best_score = max(score for _, _, score in scored)
        threshold = max(best_score * 0.45, 8.0)
        selected = [
            (index, line)
            for index, line, score in scored
            if score >= threshold
        ]
        if len(selected) < 2:
            selected = [
                (index, line)
                for index, line, _ in sorted(
                    scored,
                    key=lambda item: item[2],
                    reverse=True,
                )[:3]
            ]
        selected.sort(key=lambda item: item[0])
        return [line for _, line in selected]

    @staticmethod
    def _merge_blocks(blocks: list[dict]) -> list[tuple[str, float]]:
        blocks.sort(key=lambda block: block["cy"])
        average_height = sum(
            max(point[1] for point in block["bbox"])
            - min(point[1] for point in block["bbox"])
            for block in blocks
        ) / len(blocks)
        grouped_lines: list[list[dict]] = []
        current_line: list[dict] = []
        for block in blocks:
            current_center = (
                sum(item["cy"] for item in current_line) / len(current_line)
                if current_line
                else block["cy"]
            )
            if (
                current_line
                and abs(block["cy"] - current_center)
                >= average_height * 0.55
            ):
                grouped_lines.append(current_line)
                current_line = []
            current_line.append(block)
        if current_line:
            grouped_lines.append(current_line)

        merged_lines = []
        for group in grouped_lines:
            group.sort(key=lambda block: block["xmin"])
            text = "".join(block["text"] for block in group)
            confidence = sum(block["confidence"] for block in group) / len(group)
            if len(text) >= MRZ_LINE_MIN_LENGTH:
                merged_lines.append((text, confidence))
        return merged_lines

    def _select_best_result(
        self,
        merged_lines: list[tuple[str, float]],
        attempt: str,
    ) -> MRZResult:
        candidates = [
            *self._find_td1_candidates(merged_lines),
            *self._find_td2_candidates(merged_lines),
        ]
        if not candidates:
            return self._diagnostic_result(merged_lines)

        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        for candidate in candidates:
            if candidate.format_name == "td1":
                result = self._parse_td1_lines(
                    list(candidate.lines),
                    attempt,
                )
            else:
                result = self._parse_td2_lines(
                    (candidate.lines[0], candidate.lines[1]),
                    attempt,
                )
            if self.is_structurally_valid(result):
                return result
        return self._diagnostic_result(merged_lines)

    @classmethod
    def _find_td2_pair(
        cls,
        merged_lines: list[tuple[str, float]],
    ) -> tuple[tuple[str, float], tuple[str, float]] | None:
        candidates = cls._find_td2_candidates(merged_lines)
        if not candidates:
            return None
        best = max(candidates, key=lambda candidate: candidate.score)
        return best.lines[0], best.lines[1]

    @classmethod
    def _find_td2_candidates(
        cls,
        merged_lines: list[tuple[str, float]],
    ) -> list[MRZCandidate]:
        candidates: list[MRZCandidate] = []
        seen: set[tuple[str, str]] = set()
        for first_index in range(len(merged_lines) - 2, -1, -1):
            first = merged_lines[first_index][0]
            if not cls._has_document_header(first):
                continue
            for second_index in range(first_index + 1, len(merged_lines)):
                second = merged_lines[second_index][0]
                if cls._looks_like_td2_second_line(second):
                    pair = (
                        merged_lines[first_index],
                        merged_lines[second_index],
                    )
                    key = (pair[0][0], pair[1][0])
                    if key not in seen:
                        seen.add(key)
                        candidates.append(
                            MRZCandidate(
                                "td2",
                                pair,
                                cls._score_td2_candidate(pair, True),
                            )
                        )
        for second_index in range(1, len(merged_lines)):
            second = merged_lines[second_index][0]
            fields = cls._extract_td2_fields(second)
            if (
                not cls._looks_like_td2_second_line(second)
                or len(second) < 35
                or not is_valid_fin(fields["fin"] or "")
                or not fields["checksum_valid"]
            ):
                continue
            first = merged_lines[second_index - 1][0]
            if len(first) >= 20 and "<<" in first:
                pair = (
                    merged_lines[second_index - 1],
                    merged_lines[second_index],
                )
                key = (pair[0][0], pair[1][0])
                if key not in seen:
                    seen.add(key)
                    candidates.append(
                        MRZCandidate(
                            "td2",
                            pair,
                            cls._score_td2_candidate(pair, False),
                        )
                    )
        return candidates

    @classmethod
    def _score_td2_candidate(
        cls,
        pair: tuple[tuple[str, float], tuple[str, float]],
        has_header: bool,
    ) -> float:
        (line1, confidence1), (line2, confidence2) = pair
        fields = cls._extract_td2_fields(line2)
        checksum_valid = fields["checksum_valid"]
        composite_checksum_valid = fields["composite_checksum_valid"]
        fin_valid = is_valid_fin(fields["fin"] or "")
        serial_valid = bool(fields["checksum_valid"]) and is_valid_old_card_serial(
            fields["serial"] or ""
        )
        return (
            (30 if has_header else 10)
            + min(len(line1), TD2_LINE_LENGTH) / 3
            + min(len(line2), TD2_LINE_LENGTH) / 2
            + (20 if cls._looks_like_td2_second_line(line2) else 0)
            + (25 if checksum_valid else 0)
            + (15 if composite_checksum_valid else 0)
            + (30 if fin_valid else 0)
            + (10 if serial_valid else 0)
            + ((confidence1 + confidence2) / 2) * 20
        )

    @classmethod
    def _td2_nationality_index(cls, line: str) -> int | None:
        nationality_index = line.find(
            "AZE",
            TD2_NATIONALITY.start - 1,
            TD2_NATIONALITY.end + 2,
        )
        if nationality_index in TD2_NATIONALITY_ALLOWED:
            return nationality_index
        return None

    @classmethod
    def _looks_like_td2_second_line(cls, line: str) -> bool:
        nationality_index = cls._td2_nationality_index(line)
        if nationality_index is None or len(line) < 25:
            return False
        birth_date = correct_ocr_digits(
            line[nationality_index + 3 : nationality_index + 9]
        )
        return birth_date.isdigit()

    @classmethod
    def _extract_td2_fields(cls, line2: str) -> dict[str, object]:
        """Read older-card FIN/serial from fixed TD2 positions.

        Primary path uses the never-changing ICAO slots in ``layout.py``.
        OCR-shift recovery only runs when the fixed nationality mark moves.
        """
        normalized = cls._normalize_length(line2, TD2_LINE_LENGTH)
        nationality_index = cls._td2_nationality_index(normalized)
        if nationality_index is None:
            nationality_index = TD2_NATIONALITY.start

        # Fixed serial slot: line2[0:9] + check at [9].
        document_field = TD2_SERIAL.read(normalized)
        check_char = TD2_SERIAL_CHECK.read(normalized)
        digit_document = correct_ocr_digits(document_field)
        digit_check = correct_ocr_digits(check_char)
        checksum_valid = (
            len(digit_check) == 1
            and digit_check.isdigit()
            and compute_mrz_check_digit(digit_document)
            == int(digit_check)
        )
        if not checksum_valid and nationality_index != TD2_NATIONALITY.start:
            relative_check_index = nationality_index - 1
            relative_document = normalized[: max(relative_check_index, 0)]
            relative_check = (
                normalized[relative_check_index]
                if 0 <= relative_check_index < len(normalized)
                else ""
            )
            relative_digit_document = correct_ocr_digits(relative_document)
            relative_digit_check = correct_ocr_digits(relative_check)
            if (
                relative_digit_check.isdigit()
                and compute_mrz_check_digit(relative_digit_document)
                == int(relative_digit_check)
            ):
                document_field = relative_document
                check_char = relative_check
                digit_document = relative_digit_document
                digit_check = relative_digit_check
                checksum_valid = True

        serial = cls._extract_old_card_serial_number(
            digit_document if checksum_valid else document_field,
            digit_check if checksum_valid else check_char,
        )

        # Fixed FIN slot is line2[28:35]; if AZE shifted, keep FIN 18 after it.
        fin_start = (
            TD2_FIN.start
            if nationality_index == TD2_NATIONALITY.start
            else nationality_index + TD2_FIN_AFTER_NATIONALITY
        )
        fin = cls._recover_td2_fin(normalized, fin_start)
        canonical_fin = normalized[
            fin_start : fin_start + TD2_FIN.length
        ]
        if fin is not None and fin_start >= 0:
            normalized = (
                normalized[:fin_start]
                + fin
                + normalized[fin_start + TD2_FIN.length :]
            )
        composite_checksum_valid = cls._td2_composite_checksum_valid(
            normalized
        )
        return {
            "fin": fin,
            "serial": serial,
            "checksum_valid": checksum_valid,
            "composite_checksum_valid": composite_checksum_valid,
            "nationality_index": nationality_index,
            "fin_is_canonical": fin is not None and fin == canonical_fin,
        }

    @classmethod
    def _td2_composite_checksum_valid(cls, line2: str) -> bool:
        normalized = cls._normalize_length(line2, TD2_LINE_LENGTH)
        composite_check = correct_ocr_digits(
            TD2_COMPOSITE_CHECK.read(normalized)
        )
        composite_data = (
            normalized[0:10]
            + normalized[13:20]
            + normalized[21:35]
        )
        return (
            composite_check.isdigit()
            and compute_mrz_check_digit(composite_data)
            == int(composite_check)
        )

    @classmethod
    def _td2_composite_valid_with_fin(
        cls,
        line2: str,
        fin_start: int,
        fin: str,
    ) -> bool:
        normalized = cls._normalize_length(line2, TD2_LINE_LENGTH)
        if fin_start < 0 or fin_start + TD2_FIN.length > len(normalized):
            return False
        spliced = (
            normalized[:fin_start]
            + fin
            + normalized[fin_start + TD2_FIN.length :]
        )
        return cls._td2_composite_checksum_valid(spliced)

    @classmethod
    def _recover_td2_fin(cls, line2: str, expected_start: int) -> str | None:
        """Prefer the fixed FIN window, then nearby offsets with a check digit."""
        windows: list[tuple[int, str]] = []
        if expected_start >= 0 and expected_start + TD2_FIN.length <= len(line2):
            windows.append(
                (
                    expected_start,
                    line2[expected_start : expected_start + TD2_FIN.length],
                )
            )

        # Offset recovery only when a trailing check digit is present. This
        # avoids accepting truncated tails like "71SHORT" as a FIN.
        for offset in (-1, 1, -2, 2):
            start = expected_start + offset
            if start < 0 or start + TD2_FIN.length + 1 > len(line2):
                continue
            value = line2[start : start + TD2_FIN.length]
            if line2[start + TD2_FIN.length].isdigit():
                windows.append((start, value))

        for start, window in windows:
            resolved = resolve_fin_o0_ambiguity(
                window,
                checksum_ok=lambda fin, fin_start=start: (
                    cls._td2_composite_valid_with_fin(line2, fin_start, fin)
                ),
            )
            if resolved is not None:
                return resolved
        return None

    @staticmethod
    def _extract_old_card_serial_number(
        document_field: str,
        check_char: str,
    ) -> str | None:
        """Older Azerbaijani TD2 serials are numeric document numbers."""
        cleaned = document_field.replace("<", "")
        if not cleaned:
            return None
        digit_cleaned = correct_ocr_digits(cleaned)
        if is_valid_old_card_serial(digit_cleaned):
            digit_check = correct_ocr_digits(check_char)
            if digit_check.isdigit():
                padded = digit_cleaned.ljust(TD2_SERIAL.length, "<")[
                    : TD2_SERIAL.length
                ]
                if compute_mrz_check_digit(padded) == int(digit_check):
                    return digit_cleaned
            return digit_cleaned
        if is_valid_old_card_serial(cleaned):
            return cleaned
        return None

    @staticmethod
    def _has_document_header(line: str) -> bool:
        return bool(re.match(r"^I(?:<|A)?AZE", line))

    @staticmethod
    def _extract_new_card_serial_number(line1: str) -> str | None:
        """Read new-card serial from fixed TD1 position line1[5:14].

        Falls back to nearby starts when OCR drops/inserts a header character.
        """
        min_length = max(
            TD1_SERIAL.end,
            max(TD1_SERIAL_SEARCH_STARTS) + TD1_SERIAL.length,
        )
        normalized = line1 + ("<" * max(0, min_length - len(line1)))
        candidates: list[tuple[int, str]] = []

        for start in (TD1_SERIAL.start, *TD1_SERIAL_SEARCH_STARTS):
            chunk = normalized[start : start + TD1_SERIAL.length]
            if len(chunk) != TD1_SERIAL.length:
                continue
            if not re.fullmatch(r"A[AB][A-Z0-9]{7}", chunk):
                continue
            prefix = chunk[:2]
            raw_digits = chunk[2:]
            digits = correct_ocr_digits(raw_digits)
            serial = f"{prefix}{digits}"
            if not is_valid_new_card_serial(serial):
                continue
            corrections = sum(
                original != corrected
                for original, corrected in zip(raw_digits, digits)
            )
            score = (
                (10 - corrections)
                + (12 if start == TD1_SERIAL.start else 0)
                + (4 if start in TD1_SERIAL_SEARCH_STARTS else 0)
                + (3 if raw_digits.isdigit() else 0)
            )
            candidates.append((score, serial))

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    @classmethod
    def _has_td1_document_header(cls, line: str) -> bool:
        if cls._has_document_header(line):
            return True
        if not line.startswith("I"):
            return False
        return cls._extract_new_card_serial_number(line) is not None

    @classmethod
    def _extract_td1_fin(cls, line1: str, issuing_state: str) -> str | None:
        """Read new-card FIN from fixed optional-data start line1[15:22]."""
        normalized = cls._normalize_length(line1, TD1_LINE_LENGTH)
        candidates: list[tuple[int, str]] = []

        # Canonical optional-data window starts at fixed index 15.
        for offset in (TD1_FIN.start, TD1_FIN.start - 1, TD1_FIN.start + 1, 13, 16):
            optional_data = normalized[offset:TD1_OPTIONAL_END].rstrip("<")
            if not optional_data:
                continue
            # Common AZE layout: optional data is ``AZE`` + FIN.
            if (
                optional_data.startswith(issuing_state)
                and len(optional_data) >= len(issuing_state) + TD1_FIN.length
            ):
                value = optional_data[
                    len(issuing_state) : len(issuing_state) + TD1_FIN.length
                ]
                if is_valid_fin(value):
                    score = 20 if offset == TD1_FIN.start else 14
                    candidates.append((score, value))
            # Direct FIN at the fixed optional-data start.
            direct = optional_data[:TD1_FIN.length]
            if is_valid_fin(direct):
                # Prefer when the window is exactly FIN (no AZE prefix noise).
                exact = len(optional_data) == TD1_FIN.length
                score = (
                    (18 if exact else 10)
                    if offset == TD1_FIN.start
                    else (8 if exact else 4)
                )
                candidates.append((score, direct))

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    @classmethod
    def _looks_like_td1(cls, lines: list[tuple[str, float]]) -> bool:
        if len(lines) != 3:
            return False
        first, second, third = (line_text for line_text, _ in lines)
        return (
            cls._has_td1_document_header(first)
            and cls._looks_like_td1_second_line(second)
            and bool(third.rstrip("<"))
            and "<<" in third
        )

    @staticmethod
    def _looks_like_td1_second_line(line: str) -> bool:
        return (
            len(line) >= 20
            and sum(character.isdigit() for character in line[:20]) >= 8
        )

    @classmethod
    def _find_td1_candidates(
        cls,
        merged_lines: list[tuple[str, float]],
    ) -> list[MRZCandidate]:
        candidates: list[MRZCandidate] = []
        for first_index, first in enumerate(merged_lines):
            if not cls._has_td1_document_header(first[0]):
                continue
            for second_index in range(first_index + 1, len(merged_lines)):
                second = merged_lines[second_index]
                if not cls._looks_like_td1_second_line(second[0]):
                    continue
                for third in merged_lines[second_index + 1 :]:
                    candidate = [first, second, third]
                    if cls._looks_like_td1(candidate):
                        candidates.append(
                            MRZCandidate(
                                "td1",
                                tuple(candidate),
                                cls._score_td1_candidate(candidate),
                            )
                        )
        return candidates

    @staticmethod
    def _td1_serial_checksum_valid(line1: str) -> bool:
        serial_field = TD1_SERIAL.read(line1)
        check_digit = correct_ocr_digits(TD1_SERIAL_CHECK.read(line1))
        if len(serial_field) != TD1_SERIAL.length or not check_digit.isdigit():
            return False
        corrected_serial = (
            serial_field[:2] + correct_ocr_digits(serial_field[2:])
        )
        return compute_mrz_check_digit(corrected_serial) == int(check_digit)

    @staticmethod
    def _td1_line2_checksums_valid(line2: str) -> bool:
        normalized = MRZExtractor._normalize_length(
            line2,
            TD1_LINE_LENGTH,
        )
        birth_date = correct_ocr_digits(normalized[0:6])
        birth_check = correct_ocr_digits(normalized[6:7])
        expiry_date = correct_ocr_digits(normalized[8:14])
        expiry_check = correct_ocr_digits(normalized[14:15])
        return (
            birth_date.isdigit()
            and birth_check.isdigit()
            and compute_mrz_check_digit(birth_date) == int(birth_check)
            and expiry_date.isdigit()
            and expiry_check.isdigit()
            and compute_mrz_check_digit(expiry_date) == int(expiry_check)
        )

    @classmethod
    def _score_td1_candidate(
        cls,
        lines: list[tuple[str, float]],
    ) -> float:
        (line1, confidence1), (line2, confidence2), (
            line3,
            confidence3,
        ) = lines
        normalized_line1 = cls._normalize_length(line1, TD1_LINE_LENGTH)
        checksum_valid = cls._td1_serial_checksum_valid(normalized_line1)
        line2_checksums_valid = cls._td1_line2_checksums_valid(line2)
        nationality_index = line2.find("AZE", 12, 20)
        serial_valid = cls._extract_new_card_serial_number(line1) is not None
        return (
            (30 if cls._has_document_header(line1) else 15)
            + (10 if serial_valid else 0)
            + sum(
                max(0, 10 - abs(len(line) - TD1_LINE_LENGTH))
                for line in (line1, line2, line3)
            )
            + (15 if nationality_index in {14, 15, 16} else 5)
            + (25 if checksum_valid else 0)
            + (20 if line2_checksums_valid else 0)
            + (
                (confidence1 + confidence2 + confidence3) / 3
            )
            * 20
        )

    def _parse_td2_lines(
        self,
        pair: tuple[tuple[str, float], tuple[str, float]],
        attempt: str,
    ) -> MRZResult:
        (raw_line1, line1_confidence), (
            raw_line2,
            line2_confidence,
        ) = pair
        line1 = self._normalize_length(raw_line1, TD2_LINE_LENGTH)
        line2 = self._normalize_length(raw_line2, TD2_LINE_LENGTH)
        fields = self._extract_td2_fields(line2)
        fin = fields["fin"] if is_valid_fin(fields["fin"] or "") else None
        serial = (
            fields["serial"]
            if is_valid_old_card_serial(fields["serial"] or "")
            else None
        )
        return MRZResult(
            fin=fin,
            confidence=(line1_confidence + line2_confidence) / 2.0,
            line1=line1,
            line2=line2,
            line3="",
            checksum_valid=bool(fields["checksum_valid"]),
            method=f"td2_{attempt}" if fin else "not_found",
            card_type="older_card",
            card_serial_number=serial,
            line_confidences=(line1_confidence, line2_confidence),
            quality_score=self._score_td2_candidate(
                pair,
                self._has_document_header(raw_line1),
            ),
            fin_is_canonical=bool(fields["fin_is_canonical"]),
            secondary_checksum_valid=bool(
                fields["composite_checksum_valid"]
            ),
        )

    def _parse_td1_lines(
        self,
        selected_lines: list[tuple[str, float]],
        attempt: str,
    ) -> MRZResult:
        raw_line1, line1_confidence = selected_lines[0]
        raw_line2, line2_confidence = selected_lines[1]
        raw_line3, line3_confidence = selected_lines[2]
        line1 = self._normalize_length(raw_line1, TD1_LINE_LENGTH)
        line2 = self._normalize_length(raw_line2, TD1_LINE_LENGTH)
        line3 = self._normalize_length(raw_line3, TD1_LINE_LENGTH)

        issuing_state = TD1_ISSUING_STATE.read(line1)
        if not issuing_state.isalnum():
            issuing_state = "AZE"

        fin_candidate = self._extract_td1_fin(line1, issuing_state)
        checksum_valid = self._td1_serial_checksum_valid(line1)
        line2_checksums_valid = self._td1_line2_checksums_valid(line2)
        canonical_optional_data = line1[
            TD1_FIN.start:TD1_OPTIONAL_END
        ].rstrip("<")
        canonical_values = {
            canonical_optional_data[:TD1_FIN.length],
        }
        if canonical_optional_data.startswith(issuing_state):
            canonical_values.add(
                canonical_optional_data[
                    len(issuing_state) : len(issuing_state) + TD1_FIN.length
                ]
            )
        confidence = (
            (line1_confidence + line2_confidence + line3_confidence) / 3.0
            if all(
                value is not None
                for value in (
                    line1_confidence,
                    line2_confidence,
                    line3_confidence,
                )
            )
            else line1_confidence
        )
        is_new_card = self._looks_like_td1(selected_lines)
        fin = fin_candidate if is_new_card else None
        return MRZResult(
            fin=fin,
            confidence=confidence,
            line1=line1,
            line2=line2,
            line3=line3,
            checksum_valid=checksum_valid and is_new_card,
            method=f"td1_{attempt}" if fin else "not_found",
            card_type=(
                "new_card"
                if is_new_card
                else "unknown"
            ),
            card_serial_number=(
                self._extract_new_card_serial_number(line1)
                if is_new_card
                else None
            ),
            line_confidences=(
                line1_confidence,
                line2_confidence,
                line3_confidence,
            ),
            quality_score=self._score_td1_candidate(selected_lines),
            fin_is_canonical=fin_candidate in canonical_values,
            secondary_checksum_valid=line2_checksums_valid,
        )

    @classmethod
    def is_structurally_valid(cls, result: MRZResult) -> bool:
        if result.fin is None or not is_valid_fin(result.fin):
            return False
        if result.card_type == "new_card":
            confidences = result.line_confidences or (
                result.confidence,
                result.confidence,
                result.confidence,
            )
            lines = [
                (result.line1, confidences[0]),
                (result.line2, confidences[1]),
                (result.line3, confidences[2]),
            ]
            return cls._looks_like_td1(lines)
        if result.card_type == "older_card":
            confidences = result.line_confidences or (
                result.confidence,
                result.confidence,
            )
            lines = [
                (result.line1, confidences[0]),
                (result.line2, confidences[1]),
            ]
            return cls._find_td2_pair(lines) is not None
        return False

    @staticmethod
    def _normalize_length(value: str, expected_length: int) -> str:
        if len(value) < expected_length:
            return value + "<" * (expected_length - len(value))
        return value[:expected_length]

    @classmethod
    def _diagnostic_result(
        cls,
        merged_lines: list[tuple[str, float]],
    ) -> MRZResult:
        ranked = sorted(
            enumerate(merged_lines),
            key=lambda item: (
                item[1][0].count("<"),
                len(item[1][0]),
            ),
            reverse=True,
        )[:3]
        selected = [
            line
            for _, line in sorted(ranked, key=lambda item: item[0])
        ]
        while len(selected) < 3:
            selected.append(("", 0.0))
        confidence = (
            sum(line_confidence for _, line_confidence in selected)
            / len([text for text, _ in selected if text])
            if any(text for text, _ in selected)
            else 0.0
        )
        return MRZResult(
            fin=None,
            confidence=confidence,
            line1=selected[0][0],
            line2=selected[1][0],
            line3=selected[2][0],
            checksum_valid=False,
            method="not_found",
            card_type="unknown",
            card_serial_number=None,
        )

    @staticmethod
    def _not_found() -> MRZResult:
        return MRZResult(
            fin=None,
            confidence=0.0,
            line1="",
            line2="",
            line3="",
            checksum_valid=False,
            method="not_found",
            card_type="unknown",
            card_serial_number=None,
        )
