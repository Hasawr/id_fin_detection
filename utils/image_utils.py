import cv2
import numpy as np
from pathlib import Path

def draw_viz_debug(img: np.ndarray, label_bbox, fin_bbox, fin_text: str) -> np.ndarray:
    """
    Draw annotations on the VIZ image showing the label and value bounding boxes.
    """
    debug_img = img.copy()
    
    # Draw label bbox in blue
    if label_bbox is not None:
        pts = np.array(label_bbox, dtype=np.int32)
        cv2.polylines(debug_img, [pts], True, (255, 0, 0), 2)
        cv2.putText(debug_img, "Label", (int(pts[0][0]), int(pts[0][1] - 5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    
    # Draw FIN bbox in green
    if fin_bbox is not None:
        pts = np.array(fin_bbox, dtype=np.int32)
        cv2.polylines(debug_img, [pts], True, (0, 255, 0), 2)
        cv2.putText(debug_img, f"FIN: {fin_text}", (int(pts[0][0]), int(pts[0][1] - 5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
    return debug_img

def draw_mrz_debug(img: np.ndarray, line1: str, line2: str, line3: str, fin_text: str, id_number: str = "", card_type: str = "") -> np.ndarray:
    """
    Overlay reconstructed MRZ lines, extracted ID number, and FIN on the image.
    """
    debug_img = img.copy()
    h, w = debug_img.shape[:2]
    
    # Create a dark semi-transparent overlay at the bottom for text info
    overlay = debug_img.copy()
    cv2.rectangle(overlay, (0, h - 140), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, debug_img, 0.4, 0, debug_img)
    
    # Overlay MRZ strings & metadata
    font = cv2.FONT_HERSHEY_SIMPLEX
    if card_type:
        cv2.putText(debug_img, f"Format: {card_type}", (10, h - 120), font, 0.45, (200, 200, 200), 1)
    cv2.putText(debug_img, f"L1: {line1}", (10, h - 100), font, 0.45, (255, 255, 255), 1)
    cv2.putText(debug_img, f"L2: {line2}", (10, h - 80), font, 0.45, (255, 255, 255), 1)
    if line3:
        cv2.putText(debug_img, f"L3: {line3}", (10, h - 60), font, 0.45, (255, 255, 255), 1)
    cv2.putText(debug_img, f"ID Num: {id_number}", (10, h - 35), font, 0.55, (255, 255, 0), 2)
    cv2.putText(debug_img, f"FIN: {fin_text}", (10, h - 12), font, 0.6, (0, 255, 0), 2)
    
    return debug_img

def save_debug_image(img: np.ndarray, filename: str, output_dir: str | Path = "./debug_output") -> Path:
    """
    Save the annotated debug image to the output directory.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    full_path = out_path / filename
    cv2.imwrite(str(full_path), img)
    return full_path
