from pathlib import Path

import cv2
import numpy as np


class ImagePreprocessor:
    MRZ_HEIGHT_RATIO = 0.35
    DEFAULT_MAX_OCR_SIDE = 1600

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

    @classmethod
    def enhance_for_mrz(cls, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        mrz_crop = image[
            int(height * (1.0 - cls.MRZ_HEIGHT_RATIO)) : height,
            0:width,
        ]
        gray = cv2.cvtColor(mrz_crop, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.createCLAHE(
            clipLimit=3.0, tileGridSize=(8, 8)
        ).apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

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
        target_height = 600
        scale = min(1.0, target_height / height)
        resized = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
        )
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(
            edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        resized_area = resized.shape[0] * resized.shape[1]
        card_contour = None
        rectangular_candidate = None
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            contour_area = cv2.contourArea(contour)
            if contour_area < 0.15 * resized_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            approximate = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approximate) == 4:
                card_contour = approximate
                break
            rotated_rectangle = cv2.minAreaRect(contour)
            rectangle_width, rectangle_height = rotated_rectangle[1]
            if min(rectangle_width, rectangle_height) <= 0:
                continue
            aspect_ratio = max(rectangle_width, rectangle_height) / min(
                rectangle_width, rectangle_height
            )
            rectangularity = contour_area / (
                rectangle_width * rectangle_height
            )
            if 1.25 <= aspect_ratio <= 2.0 and rectangularity >= 0.65:
                rectangular_candidate = cv2.boxPoints(rotated_rectangle)
                break

        if card_contour is None:
            if rectangular_candidate is None:
                return image
            contour_points = rectangular_candidate
        else:
            contour_points = card_contour.reshape(4, 2)

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
