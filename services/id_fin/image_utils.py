from pathlib import Path

import cv2
import numpy as np


def draw_mrz_debug(
    image: np.ndarray, line1: str, line2: str, line3: str, fin_text: str
) -> np.ndarray:
    debug_image = image.copy()
    height, width = debug_image.shape[:2]
    overlay = debug_image.copy()
    cv2.rectangle(overlay, (0, height - 110), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, debug_image, 0.4, 0, debug_image)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(debug_image, f"L1: {line1}", (10, height - 80), font, 0.5, (255, 255, 255), 1)
    cv2.putText(debug_image, f"L2: {line2}", (10, height - 60), font, 0.5, (255, 255, 255), 1)
    cv2.putText(debug_image, f"L3: {line3}", (10, height - 40), font, 0.5, (255, 255, 255), 1)
    cv2.putText(debug_image, f"Extracted FIN: {fin_text}", (10, height - 15), font, 0.6, (0, 255, 0), 2)
    return debug_image


def save_debug_image(
    image: np.ndarray,
    filename: str,
    output_directory: str | Path = "./debug_output",
) -> Path:
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    full_path = output_path / filename
    cv2.imwrite(str(full_path), image)
    return full_path
