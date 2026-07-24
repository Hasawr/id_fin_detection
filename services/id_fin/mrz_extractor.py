from dataclasses import dataclass
import logging
import re

from . import MRZResult
from .validator import clean_mrz_line, compute_mrz_check_digit, is_valid_fin


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
        is_cropped: bool = False,
        attempt: str | None = None,
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
        attempt_name = attempt or (
            "mrz_strip" if is_cropped else "full_image"
        )
        return self._select_best_result(merged_lines, attempt_name)

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
                >= average_height * 0.7
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
            if (
                not cls._looks_like_td2_second_line(second)
                or len(second) < 35
                or not is_valid_fin(
                    cls._normalize_length(second, 36)[28:35]
                )
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
        normalized = cls._normalize_length(line2, 36)
        checksum_valid = (
            normalized[9].isdigit()
            and compute_mrz_check_digit(normalized[:9])
            == int(normalized[9])
        )
        fin_valid = is_valid_fin(normalized[28:35])
        return (
            (30 if has_header else 10)
            + min(len(line1), 36) / 3
            + min(len(line2), 36) / 2
            + (20 if cls._looks_like_td2_second_line(line2) else 0)
            + (25 if checksum_valid else 0)
            + (30 if fin_valid else 0)
            + ((confidence1 + confidence2) / 2) * 20
        )

    @staticmethod
    def _looks_like_td2_second_line(line: str) -> bool:
        nationality_index = line.find("AZE", 8, 15)
        birth_date = (
            line[nationality_index + 3 : nationality_index + 9]
            if nationality_index >= 0
            else ""
        )
        return (
            nationality_index in {9, 10, 11}
            and birth_date.isdigit()
            and len(line) >= 25
        )

    @staticmethod
    def _has_document_header(line: str) -> bool:
        return bool(re.match(r"^I(?:<|A)?AZE", line))

    @classmethod
    def _has_td1_document_header(cls, line: str) -> bool:
        serial_match = re.search(r"A[AB]\d{7}", line[:16])
        return cls._has_document_header(line) or (
            line.startswith("I")
            and serial_match is not None
            and serial_match.start() in {4, 5, 6}
        )

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
    def _find_td1_triplet(
        cls,
        merged_lines: list[tuple[str, float]],
    ) -> list[tuple[str, float]] | None:
        candidates = cls._find_td1_candidates(merged_lines)
        if not candidates:
            return None
        best = max(candidates, key=lambda candidate: candidate.score)
        return list(best.lines)

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

    @classmethod
    def _score_td1_candidate(
        cls,
        lines: list[tuple[str, float]],
    ) -> float:
        (line1, confidence1), (line2, confidence2), (
            line3,
            confidence3,
        ) = lines
        normalized_line1 = cls._normalize_length(line1, 30)
        checksum_valid = (
            normalized_line1[14].isdigit()
            and compute_mrz_check_digit(normalized_line1[5:14])
            == int(normalized_line1[14])
        )
        nationality_index = line2.find("AZE", 12, 20)
        serial_valid = cls._extract_new_card_serial_number(line1) is not None
        return (
            (30 if cls._has_document_header(line1) else 15)
            + (10 if serial_valid else 0)
            + sum(
                max(0, 10 - abs(len(line) - 30))
                for line in (line1, line2, line3)
            )
            + (15 if nationality_index in {14, 15, 16} else 5)
            + (25 if checksum_valid else 0)
            + (
                (confidence1 + confidence2 + confidence3) / 3
            )
            * 20
        )

    def _parse_td2(
        self,
        pair: tuple[tuple[str, float], tuple[str, float]],
        is_cropped: bool,
    ) -> MRZResult:
        attempt = "mrz_strip" if is_cropped else "full_image"
        return self._parse_td2_lines(pair, attempt)

    def _parse_td2_lines(
        self,
        pair: tuple[tuple[str, float], tuple[str, float]],
        attempt: str,
    ) -> MRZResult:
        (raw_line1, line1_confidence), (
            raw_line2,
            line2_confidence,
        ) = pair
        line1 = self._normalize_length(raw_line1, 36)
        line2 = self._normalize_length(raw_line2, 36)
        fin_candidate = line2[28:35]
        fin = fin_candidate if is_valid_fin(fin_candidate) else None
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
            method=f"td2_{attempt}" if fin else "not_found",
            card_type="older_card",
            card_serial_number=None,
        )

    def _parse_td1(
        self,
        merged_lines: list[tuple[str, float]],
        is_cropped: bool,
    ) -> MRZResult:
        selected_lines = self._find_td1_triplet(merged_lines)
        if selected_lines is None:
            return self._diagnostic_result(merged_lines)
        attempt = "mrz_strip" if is_cropped else "full_image"
        return self._parse_td1_lines(selected_lines, attempt)

    def _parse_td1_lines(
        self,
        selected_lines: list[tuple[str, float]],
        attempt: str,
    ) -> MRZResult:
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
                if is_valid_fin(value):
                    candidates.append(
                        (value, 10 + (2 if offset == 15 else 0), offset)
                    )
            if is_valid_fin(optional_data):
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
        fin_candidate = candidates[0][0] if candidates else None
        checksum_valid = (
            len(line1) >= 15
            and line1[14].isdigit()
            and compute_mrz_check_digit(line1[5:14]) == int(line1[14])
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
        )

    @staticmethod
    def _extract_new_card_serial_number(line1: str) -> str | None:
        match = re.search(r"A[AB]\d{7}", line1[:16])
        return match.group(0) if match is not None else None

    @classmethod
    def is_structurally_valid(cls, result: MRZResult) -> bool:
        if result.fin is None or not is_valid_fin(result.fin):
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
