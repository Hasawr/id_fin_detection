import sys
from types import SimpleNamespace

import numpy as np
import pytest

from services.id_fin import MRZResult
from services.id_fin.detector import (
    FINDetector,
    OCRAttempt,
    OCRProcessingError,
)
from services.id_fin.preprocessor import ImagePreprocessor


def test_detector_initializes_paddle_with_production_v3(monkeypatch) -> None:
    captured_options: dict[str, object] = {}

    def fake_paddle_ocr(**options):
        captured_options.update(options)
        return object()

    monkeypatch.setitem(
        sys.modules,
        "paddle",
        SimpleNamespace(
            device=SimpleNamespace(is_compiled_with_cuda=lambda: False)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        SimpleNamespace(PaddleOCR=fake_paddle_ocr),
    )

    FINDetector(use_gpu=False)

    assert captured_options["ocr_version"] == "PP-OCRv3"


def test_detector_wraps_unexpected_engine_failure() -> None:
    class FailingPreprocessor:
        @staticmethod
        def load(_image_path):
            raise RuntimeError("decode failed")

    detector = FINDetector.__new__(FINDetector)
    detector.preprocessor = FailingPreprocessor()

    with pytest.raises(OCRProcessingError, match="broken.png"):
        detector.detect_from_mrz("broken.png")


def test_detector_ranking_prefers_consensus_then_quality() -> None:
    consensus = MRZResult(
        fin="1ABC234",
        confidence=0.8,
        line1="",
        line2="",
        line3="",
        checksum_valid=False,
        method="td1_mrz_strip",
        card_type="new_card",
        card_serial_number=None,
        quality_score=50,
    )
    outlier = MRZResult(
        fin="9XYZ876",
        confidence=0.99,
        line1="",
        line2="",
        line3="",
        checksum_valid=True,
        method="td1_mrz_roi",
        card_type="new_card",
        card_serial_number=None,
        quality_score=100,
    )
    counts = {"1ABC234": 2, "9XYZ876": 1}

    assert FINDetector._result_rank(
        consensus,
        counts,
    ) > FINDetector._result_rank(outlier, counts)

    canonical = MRZResult(
        **{
            **consensus.__dict__,
            "secondary_checksum_valid": True,
            "fin_is_canonical": True,
        }
    )
    assert FINDetector._result_rank(
        canonical,
        counts,
    ) > FINDetector._result_rank(consensus, counts)


def test_detector_runs_recovery_for_single_candidate_and_picks_quality() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)

    class FakePreprocessor:
        @staticmethod
        def load(_path):
            return image

        @staticmethod
        def detect_card_roi(value):
            return value

    class FakeExtractor:
        @staticmethod
        def extract(_image, *, attempt):
            is_recovery = attempt == "full_image"
            return MRZResult(
                fin="1ABC234" if is_recovery else "9XYZ876",
                confidence=0.9,
                line1="",
                line2="",
                line3="",
                checksum_valid=True,
                method=attempt,
                card_type="new_card",
                card_serial_number=None,
                quality_score=120 if is_recovery else 80,
            )

        @staticmethod
        def is_structurally_valid(result):
            return result.fin is not None

    detector = FINDetector.__new__(FINDetector)
    detector.preprocessor = FakePreprocessor()
    detector.mrz_extractor = FakeExtractor()
    detector.save_debug_images = False
    detector._build_attempts = lambda _image: [
        OCRAttempt("mrz_strip", lambda: image),
        OCRAttempt("full_image", lambda: image),
    ]

    result = detector.detect_from_mrz("quality.png")

    assert result.fin == "1ABC234"
    assert result.mrz_result is not None
    assert result.mrz_result.method == "full_image"


def test_detector_uses_td2_line_recognition_when_detection_finds_nothing() -> None:
    image = np.zeros((40, 80, 3), dtype=np.uint8)
    line_images = [
        np.zeros((10, 80, 3), dtype=np.uint8),
        np.zeros((10, 80, 3), dtype=np.uint8),
    ]

    class FakePreprocessor:
        @staticmethod
        def load(_path):
            return image

        @staticmethod
        def detect_card_roi(value):
            return value

        @staticmethod
        def extract_td2_line_crops(_image):
            return line_images

    class FakeExtractor:
        @staticmethod
        def extract(_image, *, attempt):
            return MRZResult(
                fin=None,
                confidence=0.0,
                line1="",
                line2="",
                line3="",
                checksum_valid=False,
                method=attempt,
                card_type="unknown",
                card_serial_number=None,
            )

        @staticmethod
        def extract_recognition_lines(images, *, attempt):
            assert all(
                actual is expected
                for actual, expected in zip(images, line_images, strict=True)
            )
            return MRZResult(
                fin="3JK6ZEB",
                confidence=0.96,
                line1="I<AZEBAYRAMOV<<HATAM<<<<<<<<<<<<<<<<",
                line2="09163467<6AZE5802014M<<<<<<03JK6ZEB2",
                line3="",
                checksum_valid=True,
                method=f"td2_{attempt}",
                card_type="older_card",
                card_serial_number="09163467",
                quality_score=100,
                secondary_checksum_valid=True,
            )

        @staticmethod
        def is_structurally_valid(result):
            return result.fin is not None

    detector = FINDetector.__new__(FINDetector)
    detector.preprocessor = FakePreprocessor()
    detector.mrz_extractor = FakeExtractor()
    detector.save_debug_images = False
    detector._build_attempts = lambda _image: [
        OCRAttempt("mrz_strip", lambda: image),
    ]

    result = detector.detect_from_mrz("old-card.png")

    assert result.fin == "3JK6ZEB"
    assert result.mrz_result is not None
    assert result.mrz_result.method == "td2_line_recognition"
    assert any(
        note.startswith("OCR attempt line_recognition:")
        for note in result.notes
    )


def test_detector_retries_original_image_after_bad_card_rectification() -> None:
    original = np.zeros((80, 120, 3), dtype=np.uint8)
    bad_rectification = np.ones((30, 40, 3), dtype=np.uint8)

    class FakePreprocessor:
        @staticmethod
        def load(_path):
            return original

        @staticmethod
        def detect_card_roi(_image):
            return bad_rectification

    class FakeExtractor:
        @staticmethod
        def extract(image, *, attempt):
            is_original = image is original
            return MRZResult(
                fin="3JK6ZEB" if is_original else None,
                confidence=0.96 if is_original else 0.0,
                line1=(
                    "I<AZEBAYRAMOV<<HATAM<<<<<<<<<<<<<<<<"
                    if is_original
                    else ""
                ),
                line2=(
                    "09163467<6AZE5802014M<<<<<<03JK6ZEB2"
                    if is_original
                    else ""
                ),
                line3="",
                checksum_valid=is_original,
                method=f"td2_{attempt}" if is_original else "not_found",
                card_type="older_card" if is_original else "unknown",
                card_serial_number="09163467" if is_original else None,
                quality_score=100 if is_original else 0,
                secondary_checksum_valid=is_original,
            )

        @staticmethod
        def is_structurally_valid(result):
            return result.fin is not None

    detector = FINDetector.__new__(FINDetector)
    detector.preprocessor = FakePreprocessor()
    detector.mrz_extractor = FakeExtractor()
    detector.save_debug_images = False
    detector._build_attempts = lambda source: [
        OCRAttempt("mrz_strip", lambda: source),
    ]

    result = detector.detect_from_mrz("bad-roi.png")

    assert result.fin == "3JK6ZEB"
    assert result.mrz_result is not None
    assert result.mrz_result.method == "td2_original_mrz_strip"
    assert any(
        note.startswith("OCR attempt original_mrz_strip:")
        for note in result.notes
    )


def test_detector_stops_before_recovery_after_strong_consensus() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    attempted: list[str] = []

    class FakePreprocessor:
        @staticmethod
        def load(_path):
            return image

        @staticmethod
        def detect_card_roi(value):
            return value

    class FakeExtractor:
        @staticmethod
        def extract(_image, *, attempt):
            attempted.append(attempt)
            return MRZResult(
                fin="1ABC234",
                confidence=0.9,
                line1="",
                line2="",
                line3="",
                checksum_valid=True,
                method=attempt,
                card_type="new_card",
                card_serial_number=None,
                quality_score=100,
            )

        @staticmethod
        def is_structurally_valid(result):
            return result.fin is not None

    detector = FINDetector.__new__(FINDetector)
    detector.preprocessor = FakePreprocessor()
    detector.mrz_extractor = FakeExtractor()
    detector.save_debug_images = False
    detector._build_attempts = lambda _image: [
        OCRAttempt("mrz_strip", lambda: image),
        OCRAttempt("mrz_roi", lambda: image),
        OCRAttempt("full_image", lambda: image),
    ]

    result = detector.detect_from_mrz("consensus.png")

    assert result.fin == "1ABC234"
    assert attempted == ["mrz_strip", "mrz_roi"]


def test_detector_attempts_prioritize_grayscale_before_binarized() -> None:
    detector = FINDetector.__new__(FINDetector)
    detector.preprocessor = ImagePreprocessor()
    detector.max_ocr_side = 1600
    attempts = detector._build_attempts(
        np.zeros((200, 320, 3), dtype=np.uint8)
    )
    names = [attempt.name for attempt in attempts]

    assert names.index("mrz_strip_tight") < names.index(
        "mrz_strip_binarized"
    )
    assert names.index("mrz_strip_wide") < names.index(
        "mrz_roi_binarized"
    )


def test_detector_uses_deskew_only_after_regular_fallback_fails() -> None:
    original = np.zeros((100, 200, 3), dtype=np.uint8)
    deskewed = np.ones((100, 200, 3), dtype=np.uint8)
    attempted_names: list[str | None] = []

    class FakePreprocessor:
        MRZ_HEIGHT_RATIO = 0.35
        MRZ_HEIGHT_RATIO_TIGHT = 0.28
        MRZ_HEIGHT_RATIO_WIDE = 0.45

        @staticmethod
        def load(_path) -> np.ndarray:
            return original

        @staticmethod
        def detect_card_roi(image: np.ndarray) -> np.ndarray:
            return image

        @staticmethod
        def crop_mrz_strip(
            image: np.ndarray,
            height_ratio: float | None = None,
        ) -> np.ndarray:
            del height_ratio
            return image[:35]

        @staticmethod
        def prepare_mrz_for_ocr(
            image: np.ndarray,
            *,
            binarize: bool = False,
            adaptive_upscale: bool = False,
        ) -> np.ndarray:
            del binarize, adaptive_upscale
            return image

        @staticmethod
        def detect_mrz_roi(_image: np.ndarray) -> None:
            return None

        @staticmethod
        def bound_ocr_input(
            image: np.ndarray,
            _max_side: int,
        ) -> np.ndarray:
            return image

        @staticmethod
        def deskew(_image: np.ndarray) -> np.ndarray:
            return deskewed

        @staticmethod
        def deskew_mrz_strip(image: np.ndarray) -> np.ndarray:
            return image

    class FakeExtractor:
        @staticmethod
        def extract(
            image: np.ndarray,
            *,
            attempt: str,
        ) -> MRZResult:
            attempted_names.append(attempt)
            is_valid = image is deskewed and attempt == "deskewed_full_image"
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
    detector.save_debug_images = False

    result = detector.detect_from_mrz("tilted-card.png")

    assert result.fin == "1ABC234"
    assert result.mrz_result is not None
    assert result.mrz_result.method == "td1_deskewed_full_image"
    assert any(
        note.startswith("FIN selected from td1_deskewed_full_image")
        for note in result.notes
    )
    assert attempted_names[0] == "mrz_strip"
    assert "mrz_strip_binarized" in attempted_names
    assert attempted_names[-1] == "deskewed_full_image"
