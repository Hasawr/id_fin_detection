import numpy as np
import pytest

from services.id_fin.layout import (
    TD1_FIN,
    TD1_SERIAL,
    TD2_FIN,
    TD2_SERIAL,
)
from services.id_fin.validator import (
    clean_mrz_line,
    compute_mrz_check_digit,
    correct_ocr_digits,
    is_valid_fin,
    is_valid_new_card_serial,
    is_valid_old_card_serial,
    normalize_fin_o0,
)
from services.id_fin.mrz_extractor import MRZExtractor


def _parse_td1_for_test(
    extractor: MRZExtractor,
    lines: list[tuple[str, float]],
    *,
    is_cropped: bool,
):
    candidates = extractor._find_td1_candidates(lines)
    if not candidates:
        return extractor._diagnostic_result(lines)
    selected = max(candidates, key=lambda candidate: candidate.score)
    attempt = "mrz_strip" if is_cropped else "full_image"
    return extractor._parse_td1_lines(list(selected.lines), attempt)


def _parse_td2_for_test(
    extractor: MRZExtractor,
    pair: tuple[tuple[str, float], tuple[str, float]],
    *,
    is_cropped: bool,
):
    attempt = "mrz_strip" if is_cropped else "full_image"
    return extractor._parse_td2_lines(pair, attempt)


def test_check_digit() -> None:
    assert compute_mrz_check_digit("AB2134") == 5
    assert compute_mrz_check_digit("1234567<") == 4


def test_fin_validation() -> None:
    assert is_valid_fin("7ABC123")
    assert not is_valid_fin("TOO-LONG")


def test_ocr_digit_confusion_and_serial_validation() -> None:
    assert correct_ocr_digits("12OIZSB") == "1201258"
    assert is_valid_old_card_serial("19205792")
    assert not is_valid_old_card_serial("AA2039827")
    assert is_valid_new_card_serial("AA2039827")
    assert not is_valid_new_card_serial("19205792")


def test_fin_o0_normalization() -> None:
    # Letter O never appears in a real FIN - only digit 0 does, so the
    # correction is unconditional, unlike other OCR letter/digit mix-ups.
    assert normalize_fin_o0("2BDLLON") == "2BDLL0N"
    assert normalize_fin_o0("2BDLL0N") == "2BDLL0N"
    assert not is_valid_fin("2BDLLON")
    assert is_valid_fin(normalize_fin_o0("2BDLLON"))


def test_td1_line2_checksums_strengthen_candidate_score() -> None:
    valid = "9601226M3006162AZE<<<<<<<<<<<5"
    invalid = "9601220M3006160AZE<<<<<<<<<<<5"

    assert MRZExtractor._td1_line2_checksums_valid(valid)
    assert not MRZExtractor._td1_line2_checksums_valid(invalid)


def test_fixed_mrz_positions_for_fin_and_serial() -> None:
    new_line1 = "IAAZEAA203982795H0H4NX<<<<<<<<"
    old_line2 = "19205792<7AZE8210276M32102712BDLLON9"

    assert TD1_SERIAL.read(new_line1) == "AA2039827"
    assert TD1_FIN.read(new_line1) == "5H0H4NX"
    assert TD2_SERIAL.read(old_line2).replace("<", "") == "19205792"
    assert TD2_FIN.read(old_line2) == "2BDLLON"


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

    result = _parse_td2_for_test(extractor, pair, is_cropped=False)
    assert result.fin == "1HNLXEM"
    assert result.card_type == "older_card"
    assert result.card_serial_number == "14433235"
    assert result.checksum_valid is True
    assert result.line3 == ""


def test_td2_recognition_only_fallback_parses_visible_old_card() -> None:
    class FakeOCR:
        @staticmethod
        def ocr(images, *, det, rec, cls):
            assert len(images) == 2
            assert (det, rec, cls) == (False, True, False)
            return [
                [
                    ("I<AZEBAYRAMOV<<HATAM<<<<<<<<<<<<<<<<", 0.95),
                    ("09163467<6AZE5802014M<<<<<<03JK6ZEB2", 0.96),
                ]
            ]

    extractor = MRZExtractor(FakeOCR())
    result = extractor.extract_recognition_lines(
        [np.zeros((30, 300, 3)), np.zeros((30, 300, 3))],
        attempt="line_recognition",
    )

    assert result.fin == "3JK6ZEB"
    assert result.card_serial_number == "09163467"
    assert result.checksum_valid is True
    assert result.secondary_checksum_valid is True
    assert result.method == "td2_line_recognition"


def test_exact_new_and_old_card_mrz_layouts() -> None:
    extractor = MRZExtractor.__new__(MRZExtractor)
    new_card = _parse_td1_for_test(
        extractor,
        [
            ("IAAZEAA203982795H0H4NX<<<<<<<<", 0.99),
            ("9601226M3006162AZE<<<<<<<<<<<5", 0.99),
            ("VALIYEV<<OMAR<<<<<<<<<<<<<<<<<", 0.99),
        ],
        is_cropped=False,
    )
    old_card = _parse_td2_for_test(
        extractor,
        (
            ("I<AZEGOJAYEV<<AYKHAN<<<<<<<<<<<<<<<<", 0.99),
            ("19205792<7AZE8210276M32102712BDLLON9", 0.99),
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
    assert old_card.fin == "2BDLL0N"
    assert old_card.card_type == "older_card"
    assert old_card.card_serial_number == "19205792"
    assert len(old_card.line1) == 36
    assert len(old_card.line2) == 36
    assert old_card.line3 == ""


def test_td1_checksum_corrects_ocr_digit_confusion() -> None:
    extractor = MRZExtractor.__new__(MRZExtractor)
    result = _parse_td1_for_test(
        extractor,
        [
            ("IAAZEAA2O3982795H0H4NX<<<<<<<<", 0.99),
            ("9601226M3006162AZE<<<<<<<<<<<5", 0.99),
            ("VALIYEV<<OMAR<<<<<<<<<<<<<<<<<", 0.99),
        ],
        is_cropped=False,
    )

    assert result.card_serial_number == "AA2039827"
    assert result.checksum_valid is True


def test_td2_fin_o_reading_is_always_normalized_to_zero() -> None:
    extractor = MRZExtractor.__new__(MRZExtractor)
    # Real FIN never contains letter O - only digit 0. However OCR reads
    # that character, the extracted FIN must always come out as "2BDLL0N".
    ocr_read_digit = "19205792<7AZE8210276M32102712BDLL0N5"
    ocr_read_letter = "19205792<7AZE8210276M32102712BDLLON5"

    assert extractor._extract_td2_fields(ocr_read_digit)["fin"] == "2BDLL0N"
    assert extractor._extract_td2_fields(ocr_read_letter)["fin"] == "2BDLL0N"
    assert extractor._extract_td2_fields(ocr_read_digit)[
        "composite_checksum_valid"
    ]
    assert extractor._extract_td2_fields(ocr_read_letter)[
        "composite_checksum_valid"
    ]


def test_td2_composite_checksum_affects_candidate_score() -> None:
    extractor = MRZExtractor.__new__(MRZExtractor)
    line1 = "I<AZERASHIDOVAINARA<<<<<<<<<<<<<<<<"
    valid_line2 = "14433235<1AZE8001195F30011901HNLXEM7"
    invalid_line2 = f"{valid_line2[:-1]}0"

    valid_fields = extractor._extract_td2_fields(valid_line2)
    invalid_fields = extractor._extract_td2_fields(invalid_line2)
    valid_score = extractor._score_td2_candidate(
        ((line1, 0.99), (valid_line2, 0.99)),
        True,
    )
    invalid_score = extractor._score_td2_candidate(
        ((line1, 0.99), (invalid_line2, 0.99)),
        True,
    )

    assert valid_fields["composite_checksum_valid"] is True
    assert invalid_fields["composite_checksum_valid"] is False
    assert valid_score > invalid_score


def test_truncated_old_card_is_not_classified_as_new() -> None:
    extractor = MRZExtractor.__new__(MRZExtractor)
    merged_lines = [
        ("19<01<1980<<<<<<<<<<<<<<<<<<<<", 0.95),
        ("I<AZERASHIDOVAINARA<<<<<<<<<<", 0.98),
        ("14433235<1AZE8001195F30011901H", 0.99),
    ]

    pair = extractor._find_td2_pair(merged_lines)
    assert pair is not None
    old_result = _parse_td2_for_test(extractor, pair, is_cropped=False)
    fallback_result = _parse_td1_for_test(
        extractor,
        merged_lines,
        is_cropped=False,
    )

    assert old_result.card_type == "older_card"
    assert old_result.fin is None
    assert fallback_result.card_type == "unknown"
    assert fallback_result.fin is None


def test_structural_validation_rejects_truncated_td2() -> None:
    extractor = MRZExtractor.__new__(MRZExtractor)
    truncated_lines = [
        ("I<AZERASHIDOVAINARA<<<<<<<<<<", 0.98),
        ("14433235<1AZE8001195F3", 0.99),
    ]

    assert extractor._find_td2_pair(truncated_lines) is None


def test_structural_validation_accepts_td1_and_td2() -> None:
    extractor = MRZExtractor.__new__(MRZExtractor)
    td1 = _parse_td1_for_test(
        extractor,
        [
            ("IAAZEAA203982795H0H4NX<<<<<<<<", 0.99),
            ("9601226M3006162AZE<<<<<<<<<<<5", 0.99),
            ("VALIYEV<<OMAR<<<<<<<<<<<<<<<<<", 0.99),
        ],
        is_cropped=True,
    )
    td2 = _parse_td2_for_test(
        extractor,
        (
            ("I<AZEGOJAYEV<<AYKHAN<<<<<<<<<<<<<<<<", 0.99),
            ("19205792<7AZE8210276M32102712BDLLON9", 0.99),
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
        == "AA1230567"
    )


def test_new_card_serial_rejects_non_digit_noise_that_cannot_be_corrected() -> None:
    extractor = MRZExtractor.__new__(MRZExtractor)

    assert (
        extractor._extract_new_card_serial_number(
            "IAAZEAA123X567<<<<<<<<<<<<<<<<"
        )
        is None
    )


def test_new_card_classification_tolerates_minor_header_ocr_loss() -> None:
    extractor = MRZExtractor.__new__(MRZExtractor)
    result = _parse_td1_for_test(
        extractor,
        [
            ("IAAEAA123456701ABC234<<<<<<", 0.93),
            ("9001011M3001019A2E<<<<<<<<<<<0", 0.91),
            ("TEST<<PERSON<<<<<<<<<<<<<<<<<<", 0.92),
        ],
        is_cropped=False,
    )

    assert result.fin == "1ABC234"
    assert result.card_type == "new_card"
    assert result.card_serial_number == "AA1234567"


def test_valid_td1_fin_does_not_require_perfect_serial_ocr() -> None:
    extractor = MRZExtractor.__new__(MRZExtractor)
    result = _parse_td1_for_test(
        extractor,
        [
            ("IAAZEAA123X5670AZE1ABC234<<<<<", 0.90),
            ("9001011M3001019AZE<<<<<<<<<<<0", 0.91),
            ("TEST<<PERSON<<<<<<<<<<<<<<<<<<", 0.92),
        ],
        is_cropped=False,
    )

    assert result.fin == "1ABC234"
    assert result.card_type == "new_card"
    assert result.card_serial_number is None


def test_new_card_serial_corrects_ocr_digit_confusion() -> None:
    extractor = MRZExtractor.__new__(MRZExtractor)
    result = _parse_td1_for_test(
        extractor,
        [
            ("IAAZEAA123O5670AZE1ABC234<<<<<", 0.90),
            ("9001011M3001019AZE<<<<<<<<<<<0", 0.91),
            ("TEST<<PERSON<<<<<<<<<<<<<<<<<<", 0.92),
        ],
        is_cropped=False,
    )

    assert result.fin == "1ABC234"
    assert result.card_type == "new_card"
    assert result.card_serial_number == "AA1230567"


def test_valid_td1_tolerates_digit_confusion_in_second_line() -> None:
    extractor = MRZExtractor.__new__(MRZExtractor)
    result = _parse_td1_for_test(
        extractor,
        [
            ("IAAZEAA12345670AZE1ABC234<<<<<", 0.90),
            ("9O01011M3O01019AZE<<<<<<<<<<<0", 0.91),
            ("TEST<<PERSON<<<<<<<<<<<<<<<<<<", 0.92),
        ],
        is_cropped=False,
    )

    assert result.fin == "1ABC234"
    assert result.card_type == "new_card"
    assert result.card_serial_number == "AA1234567"


def test_td2_can_use_structural_second_line_when_header_is_lost() -> None:
    extractor = MRZExtractor.__new__(MRZExtractor)
    merged_lines = [
        ("14<05<1978<<<<<<<<<<<<<<<<<<<<", 0.97),
        ("GOJAYEV<<AYKHAN<<<<<<<<<<<<<<<<", 0.96),
        ("19205792<7AZE8210276M32102712BDLLON9", 0.98),
    ]

    pair = extractor._find_td2_pair(merged_lines)

    assert pair is not None
    result = _parse_td2_for_test(extractor, pair, is_cropped=False)
    assert result.fin == "2BDLL0N"
    assert result.card_serial_number == "19205792"
    assert result.card_type == "older_card"


def test_older_card_fin_recovers_when_aze_alignment_shifts() -> None:
    extractor = MRZExtractor.__new__(MRZExtractor)
    # Extra OCR character before nationality shifts AZE from index 10 to 11.
    result = _parse_td2_for_test(
        extractor,
        (
            ("I<AZEGOJAYEV<<AYKHAN<<<<<<<<<<<<<<<<", 0.99),
            ("19205792<7XAZE8210276M32102712BDLLON9", 0.99),
        ),
        is_cropped=False,
    )

    assert result.fin == "2BDLL0N"
    assert result.card_type == "older_card"
    assert result.card_serial_number == "19205792"


def test_older_card_tolerates_digit_confusion_in_birth_date() -> None:
    extractor = MRZExtractor.__new__(MRZExtractor)
    result = _parse_td2_for_test(
        extractor,
        (
            ("I<AZEGOJAYEV<<AYKHAN<<<<<<<<<<<<<<<<", 0.99),
            ("19205792<7AZE82IO276M32102712BDLLON9", 0.99),
        ),
        is_cropped=False,
    )

    assert result.fin == "2BDLL0N"
    assert result.card_type == "older_card"
    assert result.card_serial_number == "19205792"


def test_older_card_serial_corrects_ocr_digit_confusion() -> None:
    extractor = MRZExtractor.__new__(MRZExtractor)
    result = _parse_td2_for_test(
        extractor,
        (
            ("I<AZEGOJAYEV<<AYKHAN<<<<<<<<<<<<<<<<", 0.99),
            ("192O5792<7AZE8210276M32102712BDLLON9", 0.99),
        ),
        is_cropped=False,
    )

    assert result.fin == "2BDLL0N"
    assert result.card_serial_number == "19205792"
    assert result.checksum_valid is True


def test_strong_td1_candidate_beats_weak_headerless_td2_candidate() -> None:
    extractor = MRZExtractor.__new__(MRZExtractor)
    merged_lines = [
        ("NOISY<<NAME<<<<<<<<<<<<<<<<<<<<", 0.70),
        ("19205792<0AZE8210276M32102712BDLLON9", 0.70),
        ("IAAZEAA12345670AZE1ABC234<<<<<", 0.97),
        ("9001011M3001019AZE<<<<<<<<<<<0", 0.97),
        ("TEST<<PERSON<<<<<<<<<<<<<<<<<<", 0.97),
    ]

    result = extractor._select_best_result(
        merged_lines,
        "full_image",
    )

    assert result.fin == "1ABC234"
    assert result.card_type == "new_card"
    assert result.method == "td1_full_image"


def test_invalid_td2_candidate_does_not_block_valid_td1() -> None:
    extractor = MRZExtractor.__new__(MRZExtractor)
    merged_lines = [
        ("I<AZEOLD<<CARD<<<<<<<<<<<<<<<<<<<<", 0.85),
        ("19205792<7AZE8210276M3210271SHORT", 0.85),
        ("IAAZEAA12345670AZE1ABC234<<<<<", 0.96),
        ("9001011M3001019AZE<<<<<<<<<<<0", 0.96),
        ("TEST<<PERSON<<<<<<<<<<<<<<<<<<", 0.96),
    ]

    result = extractor._select_best_result(
        merged_lines,
        "full_image",
    )

    assert result.fin == "1ABC234"
    assert result.card_type == "new_card"


def test_unknown_td1_layout_does_not_return_label_as_fin() -> None:
    extractor = MRZExtractor.__new__(MRZExtractor)
    result = _parse_td1_for_test(
        extractor,
        [
            ("A<II<RH<QAN<QRUPU<BLOOD<GROU<<", 0.84),
            ("IAAZEAA12345670AZE1ABC234<<<<<", 0.93),
            ("TEST<<PERSON<<<<<<<<<<<<<<<<<<", 0.92),
        ],
        is_cropped=False,
    )

    assert result.card_type == "unknown"
    assert result.fin is None
    assert result.card_serial_number is None
    assert result.checksum_valid is False


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

    result = _parse_td1_for_test(
        extractor,
        merged_lines,
        is_cropped=False,
    )

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
        attempt="full_image",
    )

    assert result.fin == "1ABC234"
    assert result.card_type == "new_card"
    assert result.card_serial_number == "AA1234567"


def test_prefer_mrz_like_lines_filters_top_prose() -> None:
    extractor = MRZExtractor.__new__(MRZExtractor)
    merged_lines = [
        ("SAXSIYYAT<VASIGASININ<ETIBARLILIG<MUDDATI<BITDIKDA", 0.9),
        ("V<YA<SAXSIYYET<VESIQASI<<0<CUMLADAN", 0.9),
        ("DAYISDIRILMASI<UCUN<VASIGANI<VERAN", 0.9),
        ("IAAZEAA374226192VW8N8P<<<<<<<<", 0.95),
        ("8807179F3112157AZE<<<<<<<<<<<5", 0.94),
        ("BAHMANI<<ZIBA<<<<<<<<<<<<<<<<", 0.93),
        ("QAN<QRUPU<BLOOD<GROUP", 0.8),
    ]

    preferred = extractor._prefer_mrz_like_lines(merged_lines)

    texts = [text for text, _ in preferred]
    assert "IAAZEAA374226192VW8N8P<<<<<<<<" in texts
    assert "8807179F3112157AZE<<<<<<<<<<<5" in texts
    assert "BAHMANI<<ZIBA<<<<<<<<<<<<<<<<" in texts
    assert all(
        "ETIBARLILIG" not in text and "DAYISDIRILMASI" not in text
        for text in texts
    )
