"""
Agri-Trust | Feature Extraction Module
========================================
Uses OpenCV to quantify visual defects and color properties from crop images.

Extracted Features:
  - Bruise Detection    : Dark-region blob analysis (HSV thresholding)
  - Spot Detection      : Contour-based anomaly detection
  - Size Variance       : Bounding-rect standard deviation
  - Color Uniformity    : HSV histogram standard deviation analysis
"""

import cv2
import numpy as np
from PIL import Image
import io


class CropFeatureExtractor:
    """
    Computer-Vision pipeline for quantifying crop defects and quality
    characteristics from a single image.
    """

    def __init__(self):
        # Minimum contour area (pixels²) to qualify as a defect
        self.min_defect_area = 100
        # HSV ranges for bruise detection (dark brown / black regions)
        self.bruise_lower = np.array([0, 20, 20])
        self.bruise_upper = np.array([30, 255, 120])
        # HSV ranges for spot detection (unusual bright / discoloured patches)
        self.spot_lower = np.array([15, 40, 140])
        self.spot_upper = np.array([35, 255, 255])

    # ── Main Pipeline ─────────────────────────────────────────────────────
    def extract(self, image_bytes: bytes) -> dict:
        """
        Full defect-analysis pipeline.

        Returns:
            {
                "bruise_count": int,
                "bruise_area_pct": float,
                "spot_count": int,
                "spot_area_pct": float,
                "size_variance": float,
                "color_uniformity": float,
                "overall_defect_score": float   # 0 (perfect) → 1 (heavily defected)
            }
        """
        img = self._load_image(image_bytes)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        total_pixels = img.shape[0] * img.shape[1]

        bruise_data = self._detect_bruises(hsv, total_pixels)
        spot_data = self._detect_spots(hsv, total_pixels)
        size_var = self._compute_size_variance(img)
        color_uni = self._compute_color_uniformity(hsv)
        defect_score = self._compute_overall_score(
            bruise_data, spot_data, size_var, color_uni
        )

        return {
            "bruise_count": bruise_data["count"],
            "bruise_area_pct": round(bruise_data["area_pct"], 4),
            "spot_count": spot_data["count"],
            "spot_area_pct": round(spot_data["area_pct"], 4),
            "size_variance": round(size_var, 4),
            "color_uniformity": round(color_uni, 4),
            "overall_defect_score": round(defect_score, 4),
        }

    # ── Image Loading ─────────────────────────────────────────────────────
    @staticmethod
    def _load_image(image_bytes: bytes) -> np.ndarray:
        """Convert raw bytes to an OpenCV BGR image, resized to 512×512."""
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pil_img = pil_img.resize((512, 512))
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img

    # ── Bruise Detection ──────────────────────────────────────────────────
    def _detect_bruises(self, hsv: np.ndarray, total_pixels: int) -> dict:
        """
        Identifies dark, discoloured regions that indicate bruising.
        Uses HSV thresholding + morphological operations to isolate blobs.
        """
        mask = cv2.inRange(hsv, self.bruise_lower, self.bruise_upper)

        # Morphological cleanup: remove noise, fill small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        significant = [
            c for c in contours if cv2.contourArea(c) >= self.min_defect_area
        ]
        total_area = sum(cv2.contourArea(c) for c in significant)

        return {
            "count": len(significant),
            "area_pct": (total_area / total_pixels) * 100,
        }

    # ── Spot Detection ────────────────────────────────────────────────────
    def _detect_spots(self, hsv: np.ndarray, total_pixels: int) -> dict:
        """
        Identifies irregular bright or discoloured spots via contour analysis.
        """
        mask = cv2.inRange(hsv, self.spot_lower, self.spot_upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        significant = [
            c for c in contours if cv2.contourArea(c) >= self.min_defect_area
        ]
        total_area = sum(cv2.contourArea(c) for c in significant)

        return {
            "count": len(significant),
            "area_pct": (total_area / total_pixels) * 100,
        }

    # ── Size Variance ─────────────────────────────────────────────────────
    @staticmethod
    def _compute_size_variance(img: np.ndarray) -> float:
        """
        Measures size uniformity of objects in the image using bounding
        rectangle areas. High variance → inconsistent produce sizes.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        _, thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) < 2:
            return 0.0

        areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 50]
        if not areas:
            return 0.0

        # Normalised coefficient of variation (0 = uniform, 1 = highly varied)
        mean_area = np.mean(areas)
        if mean_area == 0:
            return 0.0
        cv_val = float(np.std(areas) / mean_area)
        return min(cv_val, 1.0)

    # ── Color Uniformity ──────────────────────────────────────────────────
    @staticmethod
    def _compute_color_uniformity(hsv: np.ndarray) -> float:
        """
        Measures how uniform the colour distribution is across the image.
        Returns 0.0 (perfectly uniform) → 1.0 (highly varied).

        Uses the standard deviation of the Hue channel histogram.
        """
        hue_channel = hsv[:, :, 0]
        hist = cv2.calcHist([hue_channel], [0], None, [180], [0, 180])
        hist = hist.flatten()
        hist_norm = hist / hist.sum() if hist.sum() > 0 else hist

        # Entropy-based uniformity
        non_zero = hist_norm[hist_norm > 0]
        entropy = -np.sum(non_zero * np.log2(non_zero))
        max_entropy = np.log2(180)  # perfectly uniform distribution

        uniformity = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
        return float(np.clip(uniformity, 0.0, 1.0))

    # ── Overall Defect Score ──────────────────────────────────────────────
    @staticmethod
    def _compute_overall_score(
        bruise_data: dict, spot_data: dict,
        size_var: float, color_uni: float
    ) -> float:
        """
        Weighted composite defect score.
        Weights: Bruises (35%), Spots (25%), Size Variance (20%), Color (20%)
        """
        bruise_norm = min(bruise_data["area_pct"] / 10.0, 1.0)
        spot_norm = min(spot_data["area_pct"] / 10.0, 1.0)

        score = (
            0.35 * bruise_norm
            + 0.25 * spot_norm
            + 0.20 * size_var
            + 0.20 * color_uni
        )
        return float(np.clip(score, 0.0, 1.0))


# ── Quick Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("✅  CropFeatureExtractor loaded. Ready for image analysis.")
