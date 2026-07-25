from pathlib import Path

import cv2
import numpy as np


class ImagePreprocessor:
    MRZ_HEIGHT_RATIO = 0.35
    MRZ_HEIGHT_RATIO_TIGHT = 0.28
    MRZ_HEIGHT_RATIO_WIDE = 0.45
    TD2_MRZ_HEIGHT_RATIO = 0.24
    DEFAULT_MAX_OCR_SIDE = 1600
    MIN_MRZ_OCR_WIDTH = 1200
    CARD_MIN_AREA_RATIO = 0.04

    @staticmethod
    def load(image_path: str | Path) -> np.ndarray:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image at path: {image_path}")
        return image

    @staticmethod
    def deskew(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        coordinates = np.column_stack(np.where(threshold > 0))
        if len(coordinates) == 0:
            return image
        angle = cv2.minAreaRect(coordinates)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        if abs(angle) <= 0.5:
            return image
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1.0)
        return cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    @staticmethod
    def deskew_mrz_strip(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        minimum_length = max(40, image.shape[1] // 4)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=40,
            minLineLength=minimum_length,
            maxLineGap=20,
        )
        if lines is None:
            return image
        angles = []
        for x1, y1, x2, y2 in lines[:, 0]:
            angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if abs(angle) <= 15:
                angles.append(angle)
        if not angles:
            return image
        angle = float(np.median(angles))
        if abs(angle) <= 0.5:
            return image
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D(
            (width / 2.0, height / 2.0),
            angle,
            1.0,
        )
        return cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    @classmethod
    def crop_mrz_strip(
        cls,
        image: np.ndarray,
        height_ratio: float | None = None,
    ) -> np.ndarray:
        ratio = cls.MRZ_HEIGHT_RATIO if height_ratio is None else height_ratio
        height, width = image.shape[:2]
        top = int(height * (1.0 - ratio))
        return image[top:height, 0:width]

    @classmethod
    def extract_td2_line_crops(
        cls,
        image: np.ndarray,
    ) -> list[np.ndarray]:
        """Locate the two bottom MRZ lines for recognition-only recovery."""
        return cls.extract_mrz_line_crops(
            image,
            line_count=2,
            height_ratio=cls.TD2_MRZ_HEIGHT_RATIO,
        )

    @classmethod
    def extract_mrz_line_crops(
        cls,
        image: np.ndarray,
        *,
        line_count: int = 2,
        height_ratio: float | None = None,
    ) -> list[np.ndarray]:
        """Locate MRZ line bands for recognition-only paths."""
        if line_count not in {2, 3}:
            raise ValueError("line_count must be 2 or 3.")
        if height_ratio == 0.0:
            # Caller already provided an MRZ strip/ROI crop.
            strip = image
        else:
            strip = cls.crop_mrz_strip(
                image,
                height_ratio=(
                    cls.TD2_MRZ_HEIGHT_RATIO
                    if height_ratio is None
                    else height_ratio
                ),
            )
        if strip.shape[0] < 20 or strip.shape[1] < 120:
            return []

        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        _, ink = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        row_ink = np.count_nonzero(ink, axis=1)
        minimum_ink = max(2, round(strip.shape[1] * 0.008))
        maximum_ink = strip.shape[1] * 0.70
        active_rows = np.flatnonzero(
            (row_ink >= minimum_ink) & (row_ink <= maximum_ink)
        )
        if active_rows.size == 0:
            return []

        bands: list[tuple[int, int]] = []
        start = end = int(active_rows[0])
        for row in active_rows[1:]:
            row = int(row)
            if row <= end + 2:
                end = row
                continue
            bands.append((start, end))
            start = end = row
        bands.append((start, end))

        minimum_height = max(3, round(strip.shape[0] * 0.04))
        text_bands = [
            (start, end)
            for start, end in bands
            if end - start + 1 >= minimum_height
        ]
        if len(text_bands) >= line_count:
            selected_bands = sorted(
                text_bands,
                key=lambda band: band[1],
            )[-line_count:]
        else:
            # Low-resolution OCR-B strokes can fragment into one-pixel rows.
            # Fall back to evenly spaced geometric bands.
            strip_height = strip.shape[0]
            selected_bands = []
            for index in range(line_count):
                top = round(strip_height * (index / line_count))
                bottom = max(
                    top + 1,
                    round(strip_height * ((index + 1) / line_count)) - 1,
                )
                selected_bands.append((top, bottom))
        padding = max(2, round(strip.shape[0] * 0.025))
        line_crops = []
        for start, end in selected_bands:
            top = max(0, start - padding)
            bottom = min(strip.shape[0], end + padding + 1)
            line = strip[top:bottom]
            enhanced = cls.enhance_mrz_image(line)
            line_crops.append(cls.upscale_mrz_if_needed(enhanced))
        return line_crops

    @classmethod
    def prepare_mrz_for_ocr(
        cls,
        image: np.ndarray,
        *,
        binarize: bool = False,
        adaptive_upscale: bool = False,
    ) -> np.ndarray:
        """Contrast + mild sharpen + optional binarize, then upscale thin MRZ crops."""
        enhanced = cls.enhance_mrz_image(image, binarize=binarize)
        min_width = (
            min(1600, max(cls.MIN_MRZ_OCR_WIDTH, image.shape[1] * 2))
            if adaptive_upscale
            else None
        )
        return cls.upscale_mrz_if_needed(enhanced, min_width=min_width)

    @staticmethod
    def enhance_mrz_image(
        image: np.ndarray,
        *,
        binarize: bool = False,
    ) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Soft glare compression keeps bright reflections from washing out MRZ.
        if int(gray.max()) - int(gray.min()) > 5:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8)).apply(gray)
        # Unsharp mask helps thin OCR-B fillers like "<".
        blurred = cv2.GaussianBlur(clahe, (0, 0), 1.2)
        sharp = cv2.addWeighted(clahe, 1.6, blurred, -0.6, 0)
        if binarize:
            binary = cv2.adaptiveThreshold(
                sharp,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                11,
            )
            # Slightly thicken strokes so Paddle keeps seeing "<".
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
            return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

    @classmethod
    def upscale_mrz_if_needed(
        cls,
        image: np.ndarray,
        min_width: int | None = None,
    ) -> np.ndarray:
        target_width = cls.MIN_MRZ_OCR_WIDTH if min_width is None else min_width
        height, width = image.shape[:2]
        if width >= target_width:
            return image
        scale = target_width / max(width, 1)
        return cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_CUBIC,
        )

    @classmethod
    def detect_mrz_roi(cls, image: np.ndarray) -> np.ndarray | None:
        height, width = image.shape[:2]
        if height < 60 or width < 120:
            return None

        scale = min(1.0, 1000 / width)
        resized = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        kernel_width = max(15, round(resized.shape[1] * 0.035))
        if kernel_width % 2 == 0:
            kernel_width += 1
        blackhat_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (kernel_width, 5),
        )
        blackhat = cv2.morphologyEx(
            gray,
            cv2.MORPH_BLACKHAT,
            blackhat_kernel,
        )
        gradient = cv2.Sobel(
            blackhat,
            ddepth=cv2.CV_32F,
            dx=1,
            dy=0,
            ksize=-1,
        )
        gradient = np.absolute(gradient)
        minimum, maximum = gradient.min(), gradient.max()
        if maximum <= minimum:
            return None
        gradient = (
            255 * (gradient - minimum) / (maximum - minimum)
        ).astype("uint8")

        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (kernel_width, 5),
        )
        connected = cv2.morphologyEx(
            gradient,
            cv2.MORPH_CLOSE,
            horizontal_kernel,
        )
        _, threshold = cv2.threshold(
            connected,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (9, max(9, round(resized.shape[0] * 0.045))),
        )
        threshold = cv2.morphologyEx(
            threshold,
            cv2.MORPH_CLOSE,
            vertical_kernel,
        )

        contours, _ = cv2.findContours(
            threshold,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        candidates: list[tuple[float, tuple[int, int, int, int]]] = []
        resized_height, resized_width = resized.shape[:2]
        for contour in contours:
            x, y, candidate_width, candidate_height = cv2.boundingRect(
                contour
            )
            if candidate_height <= 0:
                continue
            width_ratio = candidate_width / resized_width
            aspect_ratio = candidate_width / candidate_height
            bottom_position = (y + candidate_height) / resized_height
            # MRZ sits in the lower band; ignore upper prose blocks.
            if (
                width_ratio < 0.30
                or aspect_ratio < 2.2
                or bottom_position < 0.50
            ):
                continue
            score = (
                width_ratio * 3
                + min(aspect_ratio, 10) / 10
                + bottom_position * 2.5
            )
            candidates.append(
                (
                    score,
                    (x, y, candidate_width, candidate_height),
                )
            )

        if not candidates:
            return None
        _, (x, y, candidate_width, candidate_height) = max(
            candidates,
            key=lambda candidate: candidate[0],
        )
        padding_x = round(candidate_width * 0.04)
        padding_y = max(
            round(candidate_height * 0.65),
            round(resized_height * 0.025),
        )
        x1 = max(0, x - padding_x)
        y1 = max(0, y - padding_y)
        x2 = min(resized_width, x + candidate_width + padding_x)
        y2 = min(resized_height, y + candidate_height + padding_y)

        original_x1 = max(0, round(x1 / scale))
        original_y1 = max(0, round(y1 / scale))
        original_x2 = min(width, round(x2 / scale))
        original_y2 = min(height, round(y2 / scale))
        mrz_roi = image[original_y1:original_y2, original_x1:original_x2]
        return mrz_roi if mrz_roi.size else None

    @staticmethod
    def bound_ocr_input(
        image: np.ndarray,
        max_side: int = 1600,
    ) -> np.ndarray:
        if max_side <= 0:
            raise ValueError("max_side must be positive.")
        height, width = image.shape[:2]
        longest_side = max(height, width)
        if longest_side <= max_side:
            return image
        scale = max_side / longest_side
        return cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def order_points(points: np.ndarray) -> np.ndarray:
        rectangle = np.zeros((4, 2), dtype="float32")
        coordinate_sums = points.sum(axis=1)
        rectangle[0] = points[np.argmin(coordinate_sums)]
        rectangle[2] = points[np.argmax(coordinate_sums)]
        coordinate_differences = np.diff(points, axis=1)
        rectangle[1] = points[np.argmin(coordinate_differences)]
        rectangle[3] = points[np.argmax(coordinate_differences)]
        return rectangle

    @classmethod
    def detect_card_roi(cls, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        target_height = 800
        scale = min(1.0, target_height / height)
        resized = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
        )
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 40, 140)
        # Close gaps so textured backgrounds don't fragment the card outline.
        edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(
            edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        resized_area = resized.shape[0] * resized.shape[1]
        best_quad: np.ndarray | None = None
        best_score = 0.0
        best_box: np.ndarray | None = None
        best_box_score = 0.0

        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:30]:
            contour_area = cv2.contourArea(contour)
            area_ratio = contour_area / resized_area
            if area_ratio < cls.CARD_MIN_AREA_RATIO:
                continue
            perimeter = cv2.arcLength(contour, True)
            approximate = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            rotated_rectangle = cv2.minAreaRect(contour)
            rectangle_width, rectangle_height = rotated_rectangle[1]
            if min(rectangle_width, rectangle_height) <= 0:
                continue
            aspect_ratio = max(rectangle_width, rectangle_height) / min(
                rectangle_width, rectangle_height
            )
            if not 1.2 <= aspect_ratio <= 2.2:
                continue
            rectangularity = contour_area / (
                rectangle_width * rectangle_height
            )
            score = area_ratio * 2 + rectangularity
            if len(approximate) == 4 and rectangularity >= 0.55:
                if score > best_score:
                    best_score = score
                    best_quad = approximate.reshape(4, 2)
            if rectangularity >= 0.55 and score > best_box_score:
                best_box_score = score
                best_box = cv2.boxPoints(rotated_rectangle)

        if best_quad is not None:
            contour_points = best_quad
        elif best_box is not None:
            contour_points = best_box
        else:
            return image

        rectangle = cls.order_points(contour_points / scale)
        top_left, top_right, bottom_right, bottom_left = rectangle
        max_width = max(
            int(np.linalg.norm(bottom_right - bottom_left)),
            int(np.linalg.norm(top_right - top_left)),
        )
        max_height = max(
            int(np.linalg.norm(top_right - bottom_right)),
            int(np.linalg.norm(top_left - bottom_left)),
        )
        if max_width < 150 or max_height < 150:
            return image

        destination = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype="float32",
        )
        transform = cv2.getPerspectiveTransform(rectangle, destination)
        return cv2.warpPerspective(image, transform, (max_width, max_height))
