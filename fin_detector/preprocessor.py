import cv2
import numpy as np
from pathlib import Path

class ImagePreprocessor:
    """Class to load, deskew, enhance, and detect ROI from ID card images."""

    @staticmethod
    def load(image_path: str | Path) -> np.ndarray:
        """Load image via OpenCV supporting various formats."""
        path_str = str(image_path)
        img = cv2.imread(path_str)
        if img is None:
            raise FileNotFoundError(f"Could not load image at path: {path_str}")
        return img

    @staticmethod
    def deskew(img: np.ndarray) -> np.ndarray:
        """Correct card rotation/tilt using minimum bounding rectangle of text/edges."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find coordinates of all non-zero pixels
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) == 0:
            return img
            
        # Compute minimum bounding rectangle
        angle = cv2.minAreaRect(coords)[-1]
        
        # Adjust angle to correct rotation
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        # Rotate image if angle is significant
        if abs(angle) > 0.5:
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
        return img

    @staticmethod
    def enhance_for_viz(img: np.ndarray) -> np.ndarray:
        """Apply sharpening and CLAHE contrast enhancement for VIZ side."""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Convert back to BGR for OCR engine compatibility
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def enhance_for_mrz(img: np.ndarray) -> np.ndarray:
        """Crop and enhance the bottom 25% MRZ strip area."""
        h, w = img.shape[:2]
        # Crop the bottom 25%
        mrz_y = int(h * 0.75)
        mrz_crop = img[mrz_y:h, 0:w]
        
        # Enhance contrast of the cropped MRZ region
        gray = cv2.cvtColor(mrz_crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        
        # Return BGR
        return cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def enhance_for_mrz_wide(img: np.ndarray) -> np.ndarray:
        """Crop and enhance the bottom 45% MRZ area for old card (TD2) detection."""
        h, w = img.shape[:2]
        mrz_y = int(h * 0.55)
        mrz_crop = img[mrz_y:h, 0:w]
        
        gray = cv2.cvtColor(mrz_crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        
        return cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def order_points(pts: np.ndarray) -> np.ndarray:
        """Order corner points as: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype="float32")
        
        # Sum of coords: top-left has minimum sum, bottom-right has maximum sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # Difference of coords: top-right has minimum diff, bottom-left has maximum diff
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect

    @classmethod
    def detect_card_roi(cls, img: np.ndarray) -> np.ndarray:
        """Perform four-point perspective transform if a card border is detected."""
        h, w = img.shape[:2]
        # Resize for faster processing and standard contour detection
        target_h = 600
        scale = target_h / h
        resized = cv2.resize(img, (int(w * scale), target_h))
        
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Calculate total resized image area
        resized_area = resized.shape[0] * resized.shape[1]
        
        card_contour = None
        for c in contours:
            area = cv2.contourArea(c)
            # Card must occupy at least 15% of the image area
            if area < 0.15 * resized_area:
                continue
                
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            # If the contour has 4 points, we assume it's our card
            if len(approx) == 4:
                card_contour = approx
                break
                
        if card_contour is not None:
            # Rescale contour coordinates back to original size
            pts = card_contour.reshape(4, 2) / scale
            rect = cls.order_points(pts)
            (tl, tr, br, bl) = rect
            
            # Compute width of new image
            width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            max_width = max(int(width_a), int(width_b))
            
            # Compute height of new image
            height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            max_height = max(int(height_a), int(height_b))
            
            # Avoid warping to extremely small images
            if max_width < 150 or max_height < 150:
                return img
            
            # Construct destination points for rectified view
            dst = np.array([
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1]
            ], dtype="float32")
            
            # Calculate perspective transform matrix and warp image
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(img, M, (max_width, max_height))
            return warped
            
        return img
