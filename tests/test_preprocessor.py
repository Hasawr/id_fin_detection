import cv2
import numpy as np

from services.id_fin.preprocessor import ImagePreprocessor


def test_mrz_local_deskew_preserves_shape_and_rotates_lines() -> None:
    strip = np.full((120, 500, 3), 255, dtype=np.uint8)
    for y in (30, 60, 90):
        cv2.line(strip, (20, y), (480, y + 35), (0, 0, 0), 3)

    corrected = ImagePreprocessor.deskew_mrz_strip(strip)

    assert corrected.shape == strip.shape
    assert not np.array_equal(corrected, strip)


def test_mrz_strip_uses_bottom_35_percent_and_upscales() -> None:
    card = np.zeros((200, 400, 3), dtype=np.uint8)
    card[130:, :] = 255

    mrz_strip = ImagePreprocessor.crop_mrz_strip(card)
    mrz_candidate = ImagePreprocessor.prepare_mrz_for_ocr(mrz_strip)

    assert mrz_candidate.shape[1] >= ImagePreprocessor.MIN_MRZ_OCR_WIDTH
    assert abs(mrz_candidate.shape[0] / mrz_candidate.shape[1] - 70 / 400) < 0.02
    assert mrz_candidate.mean() > 200


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


def test_mrz_upscale_and_binarize_helpers() -> None:
    small = np.full((40, 200, 3), 180, dtype=np.uint8)
    cv2.putText(
        small,
        "IAAZEAA1234567",
        (5, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )

    upscaled = ImagePreprocessor.upscale_mrz_if_needed(small, min_width=400)
    binarized = ImagePreprocessor.enhance_mrz_image(small, binarize=True)

    assert upscaled.shape[1] >= 400
    assert binarized.shape[:2] == small.shape[:2]
    assert set(np.unique(binarized).tolist()).issubset({0, 255})
