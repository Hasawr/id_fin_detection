import cv2
import numpy as np

from benchmarks.benchmark_id_fin import warm_latency_improvement
from services.id_fin import FINDetectionOutput, MRZResult
from services.id_fin.preprocessor import ImagePreprocessor
from services.id_fin.service import IDFinService
from services.id_fin.validator import (
    clean_mrz_line,
    compute_mrz_check_digit,
    is_valid_fin,
)
from services.id_fin.mrz_extractor import MRZExtractor


def test_check_digit() -> None:
    assert compute_mrz_check_digit("AB2134") == 5
    assert compute_mrz_check_digit("1234567<") == 4


def test_fin_validation() -> None:
    assert is_valid_fin("7ABC123")
    assert not is_valid_fin("TOO-LONG")


def test_mrz_cleanup() -> None:
    assert clean_mrz_line("ID AZE,123") == "ID<AZE<123"
    assert clean_mrz_line("AZE$123") == "AZE<123"


def test_old_card_td2_fin_extraction() -> None:
    extractor = MRZExtractor.__new__(MRZExtractor)
    merged_lines = [
        ("19<01<1980<<<<<<<<<<<<<<<<<<<<", 0.95),
        ("I<AZERASHIDOVAINARA<<<<<<<<<<", 0.98),
        ("14433235<1AZE8001195F30011901HNLXEM7", 0.99),
    ]

    pair = extractor._find_td2_pair(merged_lines)
    assert pair is not None

    result = extractor._parse_td2(pair, is_cropped=False)
    assert result.fin == "1HNLXEM"
    assert result.card_type == "older_card"
    assert result.checksum_valid is True
    assert result.line3 == ""


def test_exact_new_and_old_card_mrz_layouts() -> None:
    extractor = MRZExtractor.__new__(MRZExtractor)
    new_card = extractor._parse_td1(
        [
            ("IAAZEAA203982795H0H4NX<<<<<<<<", 0.99),
            ("9601226M3006162AZE<<<<<<<<<<<5", 0.99),
            ("VALIYEV<<OMAR<<<<<<<<<<<<<<<<<", 0.99),
        ],
        is_cropped=False,
    )
    old_card = extractor._parse_td2(
        (
            ("I<AZEGOJAYEV<<AYKHAN<<<<<<<<<<<<<<<<", 0.99),
            ("19205792<7AZE8210276M32102712BDLLON5", 0.99),
        ),
        is_cropped=False,
    )

    assert new_card.fin == "5H0H4NX"
    assert new_card.card_type == "new_card"
    assert new_card.card_serial_number == "AA2039827"
    assert all(
        len(line) == 30
        for line in (new_card.line1, new_card.line2, new_card.line3)
    )
    assert old_card.fin == "2BDLLON"
    assert old_card.card_type == "older_card"
    assert old_card.card_serial_number is None
    assert len(old_card.line1) == 36
    assert len(old_card.line2) == 36
    assert old_card.line3 == ""


def test_truncated_old_card_is_not_classified_as_new() -> None:
    extractor = MRZExtractor.__new__(MRZExtractor)
    merged_lines = [
        ("19<01<1980<<<<<<<<<<<<<<<<<<<<", 0.95),
        ("I<AZERASHIDOVAINARA<<<<<<<<<<", 0.98),
        ("14433235<1AZE8001195F30011901H", 0.99),
    ]

    pair = extractor._find_td2_pair(merged_lines)
    assert pair is not None
    old_result = extractor._parse_td2(pair, is_cropped=False)
    fallback_result = extractor._parse_td1(merged_lines, is_cropped=False)

    assert old_result.card_type == "older_card"
    assert old_result.fin is None
    assert fallback_result.card_type == "unknown"


def test_localized_card_uses_bottom_35_percent_for_mrz() -> None:
    card = np.zeros((200, 400, 3), dtype=np.uint8)
    card[130:, :] = 255

    mrz_candidate = ImagePreprocessor.enhance_for_mrz(card)

    assert mrz_candidate.shape == (70, 400, 3)
    assert mrz_candidate.mean() > 250


def test_conservative_card_localization_and_input_cap() -> None:
    photo = np.full((800, 1000, 3), 255, dtype=np.uint8)
    cv2.rectangle(photo, (120, 180), (880, 660), (30, 30, 30), 8)

    localized = ImagePreprocessor.detect_card_roi(photo)
    bounded = ImagePreprocessor.bound_ocr_input(photo, max_side=500)

    assert localized is not photo
    assert 1.4 < localized.shape[1] / localized.shape[0] < 1.8
    assert max(bounded.shape[:2]) == 500
    assert bounded.shape[1] / bounded.shape[0] == photo.shape[1] / photo.shape[0]


def test_structural_validation_rejects_truncated_td2() -> None:
    extractor = MRZExtractor.__new__(MRZExtractor)
    truncated_lines = [
        ("I<AZERASHIDOVAINARA<<<<<<<<<<", 0.98),
        ("14433235<1AZE8001195F3", 0.99),
    ]

    assert extractor._find_td2_pair(truncated_lines) is None


def test_structural_validation_accepts_td1_and_td2() -> None:
    extractor = MRZExtractor.__new__(MRZExtractor)
    td1 = extractor._parse_td1(
        [
            ("IAAZEAA203982795H0H4NX<<<<<<<<", 0.99),
            ("9601226M3006162AZE<<<<<<<<<<<5", 0.99),
            ("VALIYEV<<OMAR<<<<<<<<<<<<<<<<<", 0.99),
        ],
        is_cropped=True,
    )
    td2 = extractor._parse_td2(
        (
            ("I<AZEGOJAYEV<<AYKHAN<<<<<<<<<<<<<<<<", 0.99),
            ("19205792<7AZE8210276M32102712BDLLON5", 0.99),
        ),
        is_cropped=True,
    )

    assert extractor.is_structurally_valid(td1)
    assert extractor.is_structurally_valid(td2)


def test_new_card_serial_number_requires_aa_or_ab_and_seven_digits() -> None:
    extractor = MRZExtractor.__new__(MRZExtractor)

    assert (
        extractor._extract_new_card_serial_number(
            "IAAZEAB1234567<<<<<<<<<<<<<<<<"
        )
        == "AB1234567"
    )
    assert (
        extractor._extract_new_card_serial_number(
            "IAAZEAC1234567<<<<<<<<<<<<<<<<"
        )
        is None
    )
    assert (
        extractor._extract_new_card_serial_number(
            "IAAZEAA123O567<<<<<<<<<<<<<<<<"
        )
        is None
    )


def test_td1_selection_ignores_interleaved_non_mrz_text() -> None:
    extractor = MRZExtractor.__new__(MRZExtractor)
    merged_lines = [
        ("AZORBAYCAN", 0.99),
        ("IAAZEAA12345670AZE1ABC234<<<<<", 0.93),
        ("VSIQANIN<NOMRASI<CARD<NO", 0.74),
        ("9001011M3001019AZE<<<<<<<<<<<0", 0.89),
        ("AA1234567", 0.99),
        ("TEST<<PERSON<<<<<<<<<<<<<<<<<<", 0.87),
        ("ETIBARLILIQ<MUDDATI", 0.84),
    ]

    result = extractor._parse_td1(merged_lines, is_cropped=False)

    assert result.fin == "1ABC234"
    assert result.card_type == "new_card"
    assert result.card_serial_number == "AA1234567"


def test_ocr_grouping_does_not_chain_adjacent_rows() -> None:
    def make_block(
        text: str,
        center_y: int,
        left: int,
    ) -> list[object]:
        return [
            [
                [left, center_y - 10],
                [left + 200, center_y - 10],
                [left + 200, center_y + 10],
                [left, center_y + 10],
            ],
            (text, 0.95),
        ]

    class FakeOCR:
        @staticmethod
        def ocr(*_args, **_kwargs) -> list[list[list[object]]]:
            return [
                [
                    make_block(
                        "IAAZEAA12345670AZE1ABC234<<<<<",
                        40,
                        10,
                    ),
                    make_block("CARD NO", 48, 600),
                    make_block(
                        "9001011M3001019AZE<<<<<<<<<<<0",
                        60,
                        10,
                    ),
                    make_block("OTHER TEXT", 62, 800),
                    make_block("AA1234567", 76, 600),
                    make_block(
                        "TEST<<PERSON<<<<<<<<<<<<<<<<<<",
                        80,
                        10,
                    ),
                ]
            ]

    extractor = MRZExtractor(FakeOCR())
    result = extractor.extract(
        np.zeros((100, 1000, 3), dtype=np.uint8),
        is_cropped=False,
    )

    assert result.fin == "1ABC234"
    assert result.card_type == "new_card"
    assert result.card_serial_number == "AA1234567"


def test_service_serializes_card_serial_number() -> None:
    mrz_result = MRZResult(
        fin="1ABC234",
        confidence=0.95,
        line1="IAAZEAA12345670AZE1ABC234<<<<<",
        line2="9001011M3001019AZE<<<<<<<<<<<0",
        line3="TEST<<PERSON<<<<<<<<<<<<<<<<<<",
        checksum_valid=True,
        method="mrz_strip",
        card_type="new_card",
        card_serial_number="AA1234567",
    )

    serialized = IDFinService._serialize(
        FINDetectionOutput(
            fin="1ABC234",
            confidence=0.95,
            mrz_result=mrz_result,
        )
    )

    mrz_details = serialized["mrz_details"]
    assert isinstance(mrz_details, dict)
    assert mrz_details["card_serial_number"] == "AA1234567"


def test_warm_latency_gate_requires_20_percent_improvement() -> None:
    assert warm_latency_improvement(0.08, 0.10) >= 0.2
    assert warm_latency_improvement(0.081, 0.10) < 0.2
