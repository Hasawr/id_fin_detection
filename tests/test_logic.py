import cv2
import numpy as np

from benchmarks.benchmark_id_fin import warm_latency_improvement
from services.id_fin import FINDetectionOutput, MRZResult
from services.id_fin.detector import FINDetector
from services.id_fin.preprocessor import ImagePreprocessor
from services.id_fin.service import serialize_fin_detection
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
)
from services.id_fin.mrz_extractor import MRZExtractor


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


def test_fixed_mrz_positions_for_fin_and_serial() -> None:
    new_line1 = "IAAZEAA203982795H0H4NX<<<<<<<<"
    old_line2 = "19205792<7AZE8210276M32102712BDLLON5"

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

    result = extractor._parse_td2(pair, is_cropped=False)
    assert result.fin == "1HNLXEM"
    assert result.card_type == "older_card"
    assert result.card_serial_number == "14433235"
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
    assert old_card.card_serial_number == "19205792"
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
    assert fallback_result.fin is None


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


def test_mrz_roi_localizes_wide_text_block() -> None:
    card = np.full((500, 800, 3), 245, dtype=np.uint8)
    for index, text in enumerate(
        (
            "IAAZEAA12345670AZE1ABC234<<<<<",
            "9001011M3001019AZE<<<<<<<<<<<0",
            "TEST<<PERSON<<<<<<<<<<<<<<<<<<",
        )
    ):
        cv2.putText(
            card,
            text,
            (60, 360 + index * 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (10, 10, 10),
            2,
            cv2.LINE_AA,
        )

    mrz_roi = ImagePreprocessor.detect_mrz_roi(card)

    assert mrz_roi is not None
    assert mrz_roi.shape[0] < card.shape[0] * 0.6
    assert mrz_roi.shape[1] > card.shape[1] * 0.5


def test_mrz_roi_returns_none_without_text_structure() -> None:
    blank = np.full((500, 800, 3), 245, dtype=np.uint8)

    assert ImagePreprocessor.detect_mrz_roi(blank) is None


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
    result = extractor._parse_td1(
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
    result = extractor._parse_td1(
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
    result = extractor._parse_td1(
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
    result = extractor._parse_td1(
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
        ("19205792<7AZE8210276M32102712BDLLON5", 0.98),
    ]

    pair = extractor._find_td2_pair(merged_lines)

    assert pair is not None
    result = extractor._parse_td2(pair, is_cropped=False)
    assert result.fin == "2BDLLON"
    assert result.card_serial_number == "19205792"
    assert result.card_type == "older_card"


def test_older_card_fin_recovers_when_aze_alignment_shifts() -> None:
    extractor = MRZExtractor.__new__(MRZExtractor)
    # Extra OCR character before nationality shifts AZE from index 10 to 11.
    result = extractor._parse_td2(
        (
            ("I<AZEGOJAYEV<<AYKHAN<<<<<<<<<<<<<<<<", 0.99),
            ("19205792<7XAZE8210276M32102712BDLLON5", 0.99),
        ),
        is_cropped=False,
    )

    assert result.fin == "2BDLLON"
    assert result.card_type == "older_card"
    assert result.card_serial_number == "19205792"


def test_older_card_tolerates_digit_confusion_in_birth_date() -> None:
    extractor = MRZExtractor.__new__(MRZExtractor)
    result = extractor._parse_td2(
        (
            ("I<AZEGOJAYEV<<AYKHAN<<<<<<<<<<<<<<<<", 0.99),
            ("19205792<7AZE82IO276M32102712BDLLON5", 0.99),
        ),
        is_cropped=False,
    )

    assert result.fin == "2BDLLON"
    assert result.card_type == "older_card"
    assert result.card_serial_number == "19205792"


def test_older_card_serial_corrects_ocr_digit_confusion() -> None:
    extractor = MRZExtractor.__new__(MRZExtractor)
    result = extractor._parse_td2(
        (
            ("I<AZEGOJAYEV<<AYKHAN<<<<<<<<<<<<<<<<", 0.99),
            ("192O5792<7AZE8210276M32102712BDLLON5", 0.99),
        ),
        is_cropped=False,
    )

    assert result.fin == "2BDLLON"
    assert result.card_serial_number == "19205792"
    assert result.checksum_valid is True


def test_strong_td1_candidate_beats_weak_headerless_td2_candidate() -> None:
    extractor = MRZExtractor.__new__(MRZExtractor)
    merged_lines = [
        ("NOISY<<NAME<<<<<<<<<<<<<<<<<<<<", 0.70),
        ("19205792<0AZE8210276M32102712BDLLON5", 0.70),
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
    result = extractor._parse_td1(
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

    serialized = serialize_fin_detection(
        FINDetectionOutput(
            fin="1ABC234",
            confidence=0.95,
            mrz_result=mrz_result,
        )
    )

    mrz_details = serialized["mrz_details"]
    assert isinstance(mrz_details, dict)
    assert mrz_details["card_serial_number"] == "AA1234567"


def test_detector_uses_deskew_only_after_regular_fallback_fails() -> None:
    original = np.zeros((100, 200, 3), dtype=np.uint8)
    deskewed = np.ones((100, 200, 3), dtype=np.uint8)
    attempted_names: list[str | None] = []

    class FakePreprocessor:
        @staticmethod
        def load(_path) -> np.ndarray:
            return original

        @staticmethod
        def detect_card_roi(image: np.ndarray) -> np.ndarray:
            return image

        @staticmethod
        def enhance_for_mrz(image: np.ndarray) -> np.ndarray:
            return image[:35]

        @staticmethod
        def detect_mrz_roi(_image: np.ndarray) -> None:
            return None

        @staticmethod
        def enhance_mrz_image(image: np.ndarray) -> np.ndarray:
            return image

        @staticmethod
        def bound_ocr_input(
            image: np.ndarray,
            _max_side: int,
        ) -> np.ndarray:
            return image

        @staticmethod
        def deskew(_image: np.ndarray) -> np.ndarray:
            return deskewed

    class FakeExtractor:
        @staticmethod
        def extract(
            image: np.ndarray,
            is_cropped: bool,
            attempt: str | None = None,
        ) -> MRZResult:
            attempted_names.append(attempt)
            is_valid = image is deskewed and not is_cropped
            return MRZResult(
                fin="1ABC234" if is_valid else None,
                confidence=0.95 if is_valid else 0.0,
                line1="IAAZEAA12345670AZE1ABC234<<<<<" if is_valid else "",
                line2="9001011M3001019AZE<<<<<<<<<<<0" if is_valid else "",
                line3="TEST<<PERSON<<<<<<<<<<<<<<<<<<" if is_valid else "",
                checksum_valid=is_valid,
                method=f"td1_{attempt}" if is_valid else "not_found",
                card_type="new_card" if is_valid else "unknown",
                card_serial_number="AA1234567" if is_valid else None,
            )

        @staticmethod
        def is_structurally_valid(result: MRZResult) -> bool:
            return result.fin is not None

    detector = FINDetector.__new__(FINDetector)
    detector.preprocessor = FakePreprocessor()
    detector.mrz_extractor = FakeExtractor()
    detector.max_ocr_side = 1600
    detector.debug = False

    result = detector.detect_from_mrz("tilted-card.png")

    assert result.fin == "1ABC234"
    assert result.mrz_result is not None
    assert result.mrz_result.method == "td1_deskewed_full_image"
    assert (
        "FIN extracted using deskewed_full_image."
        in result.notes
    )
    assert attempted_names == [
        "mrz_strip",
        "deskewed_mrz_strip",
        "full_image",
        "deskewed_full_image",
    ]


def test_warm_latency_gate_requires_20_percent_improvement() -> None:
    assert warm_latency_improvement(0.08, 0.10) >= 0.2
    assert warm_latency_improvement(0.081, 0.10) < 0.2
