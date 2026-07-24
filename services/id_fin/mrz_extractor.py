import logging
import re

from . import MRZResult
from .validator import clean_mrz_line, compute_mrz_check_digit


logger = logging.getLogger(__name__)
MRZ_LINE_MIN_LENGTH = 10


class MRZExtractor:
    def __init__(self, ocr_engine):
        self.ocr = ocr_engine

    def extract(self, image, is_cropped: bool = False) -> MRZResult:
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

        blocks.sort(key=lambda block: block["cy"])
        average_height = sum(
            max(point[1] for point in block["bbox"])
            - min(point[1] for point in block["bbox"])
            for block in blocks
        ) / len(blocks)
        grouped_lines: list[list[dict]] = []
        current_line: list[dict] = []
        for block in blocks:
            if current_line and abs(block["cy"] - current_line[-1]["cy"]) >= average_height * 0.7:
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
        td2_pair = self._find_td2_pair(merged_lines)
        if td2_pair is not None:
            return self._parse_td2(td2_pair, is_cropped)
        return self._parse_td1(merged_lines, is_cropped)

    @staticmethod
    def _find_td2_pair(
        merged_lines: list[tuple[str, float]],
    ) -> tuple[tuple[str, float], tuple[str, float]] | None:
        for first_index in range(len(merged_lines) - 2, -1, -1):
            first = merged_lines[first_index][0]
            if not MRZExtractor._has_document_header(first):
                continue
            for second_index in range(first_index + 1, len(merged_lines)):
                second = merged_lines[second_index][0]
                nationality_index = second.find("AZE", 8, 15)
                birth_date = (
                    second[nationality_index + 3 : nationality_index + 9]
                    if nationality_index >= 0
                    else ""
                )
                if (
                    nationality_index in {9, 10, 11}
                    and birth_date.isdigit()
                    and len(second) >= 25
                ):
                    return (
                        merged_lines[first_index],
                        merged_lines[second_index],
                    )
        return None

    @staticmethod
    def _has_document_header(line: str) -> bool:
        return bool(re.match(r"^I(?:<|A)?AZE", line))

    @staticmethod
    def _looks_like_td1(lines: list[tuple[str, float]]) -> bool:
        if len(lines) != 3:
            return False
        first, second, third = (line_text for line_text, _ in lines)
        nationality_index = second.find("AZE", 13, 20)
        return (
            MRZExtractor._has_document_header(first)
            and nationality_index in {14, 15, 16}
            and second[:6].isdigit()
            and bool(third.rstrip("<"))
        )

    def _parse_td2(
        self,
        pair: tuple[tuple[str, float], tuple[str, float]],
        is_cropped: bool,
    ) -> MRZResult:
        (raw_line1, line1_confidence), (
            raw_line2,
            line2_confidence,
        ) = pair
        line1 = self._normalize_length(raw_line1, 36)
        line2 = self._normalize_length(raw_line2, 36)
        fin_candidate = line2[28:35]
        fin = (
            fin_candidate
            if re.fullmatch(r"[A-Z0-9]{7}", fin_candidate)
            else None
        )
        checksum_valid = (
            line2[9].isdigit()
            and compute_mrz_check_digit(line2[0:9]) == int(line2[9])
        )
        return MRZResult(
            fin=fin,
            confidence=(line1_confidence + line2_confidence) / 2.0,
            line1=line1,
            line2=line2,
            line3="",
            checksum_valid=checksum_valid,
            method=(
                "td2_mrz_strip"
                if fin and is_cropped
                else "td2_full_image_fallback"
                if fin
                else "td2_not_found"
            ),
            card_type="older_card",
        )

    def _parse_td1(
        self,
        merged_lines: list[tuple[str, float]],
        is_cropped: bool,
    ) -> MRZResult:
        selected_lines = merged_lines[-3:]
        while len(selected_lines) < 3:
            selected_lines.append(("", 0.0))

        raw_line1, line1_confidence = selected_lines[0]
        raw_line2, line2_confidence = selected_lines[1]
        raw_line3, line3_confidence = selected_lines[2]
        line1 = self._normalize_length(raw_line1, 30)
        line2 = self._normalize_length(raw_line2, 30)
        line3 = self._normalize_length(raw_line3, 30)

        issuing_state = line1[2:5]
        if not issuing_state.isalnum():
            issuing_state = "AZE"

        candidates: list[tuple[str, int, int]] = []
        for offset in [15, 14, 13, 16]:
            optional_data = line1[offset:29].rstrip("<")
            if not optional_data:
                continue
            if (
                optional_data.startswith(issuing_state)
                and len(optional_data) >= len(issuing_state) + 7
            ):
                value = optional_data[
                    len(issuing_state) : len(issuing_state) + 7
                ]
                if re.fullmatch(r"[A-Z0-9]{7}", value):
                    candidates.append(
                        (value, 10 + (2 if offset == 15 else 0), offset)
                    )
            if re.fullmatch(r"[A-Z0-9]{7}", optional_data):
                candidates.append(
                    (optional_data, 8 + (2 if offset == 15 else 0), offset)
                )
            for value in re.findall(r"[A-Z0-9]{7}", optional_data):
                if value != issuing_state:
                    candidates.append((value, 5, offset))
            for value in re.findall(r"[A-Z0-9]{5,8}", optional_data):
                if value != issuing_state:
                    candidates.append((value, 3, offset))

        candidates.sort(key=lambda item: item[0])
        candidates.sort(key=lambda item: item[1], reverse=True)
        fin = candidates[0][0] if candidates else None
        checksum_valid = (
            len(line1) >= 15
            and line1[14].isdigit()
            and compute_mrz_check_digit(line1[5:14]) == int(line1[14])
        )
        confidence = (
            (line1_confidence + line2_confidence + line3_confidence) / 3.0
            if line1_confidence and line2_confidence and line3_confidence
            else line1_confidence
        )
        return MRZResult(
            fin=fin,
            confidence=confidence,
            line1=line1,
            line2=line2,
            line3=line3,
            checksum_valid=checksum_valid,
            method=(
                "mrz_strip"
                if fin and is_cropped
                else "full_image_fallback"
                if fin
                else "not_found"
            ),
            card_type=(
                "new_card"
                if self._looks_like_td1(selected_lines)
                else "unknown"
            ),
        )

    @classmethod
    def is_structurally_valid(cls, result: MRZResult) -> bool:
        if result.fin is None or not re.fullmatch(
            r"[A-Z0-9]{7}", result.fin
        ):
            return False
        if result.card_type == "new_card":
            lines = [
                (result.line1, result.confidence),
                (result.line2, result.confidence),
                (result.line3, result.confidence),
            ]
            return cls._looks_like_td1(lines)
        if result.card_type == "older_card":
            lines = [
                (result.line1, result.confidence),
                (result.line2, result.confidence),
            ]
            return cls._find_td2_pair(lines) is not None
        return False

    @staticmethod
    def _normalize_length(value: str, expected_length: int) -> str:
        if len(value) < expected_length:
            return value + "<" * (expected_length - len(value))
        return value[:expected_length]

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
        )
