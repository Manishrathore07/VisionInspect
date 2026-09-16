"""
Anomaly Detection and Hybrid Multi-Paradigm Fusion Module.
Fuses classical morphological gradients with statistical texture anomalies and classifies defect types.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np

from vision_inspect.config import InspectionConfig
from vision_inspect.classical_engine import ClassicalDetectionResult
from vision_inspect.feature_extractor import TextureFeatureExtractor, TextureFeatures


@dataclass
class ScoredDefect:
    """Individual detected defect candidate with spatial and statistical scores."""
    defect_id: int
    defect_type: str  # 'crack', 'scratch', 'pitting', 'blemish'
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    contour: np.ndarray
    area: float
    aspect_ratio: float
    extent: float
    solidity: float
    mean_intensity_drop: float
    texture_features: TextureFeatures


@dataclass
class AnomalyResult:
    """Consolidated defect detection output."""
    fused_binary_mask: np.ndarray
    anomaly_heatmap: np.ndarray
    detected_defects: List[ScoredDefect]
    overall_anomaly_score: float


class AnomalyDetector:
    """
    Fuses classical contour extractions with Z-score localized texture anomalies
    and performs geometric shape typing (cracks vs pitting vs scratches).
    """

    def __init__(self, config: Optional[InspectionConfig] = None):
        self.config = config or InspectionConfig()
        self.texture_extractor = TextureFeatureExtractor(self.config.texture)

    def classify_defect_geometry(
        self,
        contour: np.ndarray,
        patch: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> Tuple[str, float]:
        """
        Classifies defect type using morphological shape factors (aspect ratio, solidity, extent).
        """
        x, y, w, h = bbox
        area = cv2.contourArea(contour)
        if area <= 0:
            area = max(1.0, float(w * h))

        # Compute oriented bounding box (independent of rotation)
        rect = cv2.minAreaRect(contour)
        (rw, rh) = rect[1]
        oriented_aspect_ratio = max(rw / max(1.0, rh), rh / max(1.0, rw))

        # Thinness / Circularity metric: 4 * pi * Area / (Perimeter^2)
        perimeter = cv2.arcLength(contour, True)
        circularity = float((4.0 * np.pi * area) / (perimeter ** 2)) if perimeter > 0 else 1.0

        # Bounding box extent (area / rect_area)
        rect_area = float(w * h)
        extent = float(area / rect_area) if rect_area > 0 else 0.0

        # Convex hull solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area / hull_area) if hull_area > 0 else 1.0

        # Classification decision rules
        # Cracks: very high perimeter-to-area (low circularity < 0.20) or low solidity (< 0.50)
        if circularity < 0.20 or solidity < 0.55 or oriented_aspect_ratio >= self.config.severity.crack_aspect_ratio_threshold:
            if oriented_aspect_ratio >= 3.0 and circularity >= 0.15:
                defect_type = "scratch"
                confidence = min(0.96, 0.70 + 0.04 * oriented_aspect_ratio)
            else:
                defect_type = "crack"
                confidence = min(0.98, 0.75 + 0.20 * (1.0 - circularity))
        elif circularity > 0.60 and solidity > 0.80:
            defect_type = "pitting"
            confidence = min(0.95, 0.75 + 0.20 * circularity)
        else:
            defect_type = "blemish"
            confidence = 0.75

        return defect_type, float(confidence)

    def compute_anomaly_heatmap(self, image: np.ndarray, classical_mask: np.ndarray) -> np.ndarray:
        """
        Generates a continuous 0.0 - 1.0 normalized anomaly probability heatmap
        by combining smoothed classical mask and gradient energy.
        """
        # Distance transform to propagate gradient significance outward
        dist = cv2.distanceTransform(cv2.bitwise_not(classical_mask), cv2.DIST_L2, 5)
        # Invert so defect centers are high
        dist_inv = np.clip(1.0 - (dist / 25.0), 0.0, 1.0)

        # Smooth to create continuous thermal distribution
        blurred = cv2.GaussianBlur(dist_inv, (15, 15), 0)
        return (blurred * 255).astype(np.uint8)

    def detect(
        self,
        enhanced_image: np.ndarray,
        classical_result: ClassicalDetectionResult
    ) -> AnomalyResult:
        """
        Fuses candidate contours with statistical texture scoring and shapes.
        """
        scored_defects: List[ScoredDefect] = []
        fused_mask = np.zeros_like(classical_result.binary_mask)

        for i, cnt in enumerate(classical_result.candidate_contours):
            area = float(cv2.contourArea(cnt))
            x, y, w, h = cv2.boundingRect(cnt)

            # Extract local patch for texture evaluation
            patch = enhanced_image[y:y + h, x:x + w]
            texture_feats = self.texture_extractor.compute_patch_features(patch)

            # Classify geometry
            defect_type, confidence = self.classify_defect_geometry(cnt, patch, (x, y, w, h))

            aspect_ratio = max(w / max(1, h), h / max(1, w))
            rect_area = float(w * h)
            extent = float(area / rect_area) if rect_area > 0 else 0.0

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area / hull_area) if hull_area > 0 else 1.0

            intensity_drop = float(np.mean(enhanced_image) - (np.mean(patch) if patch.size > 0 else 0))

            defect = ScoredDefect(
                defect_id=i + 1,
                defect_type=defect_type,
                confidence=round(confidence, 3),
                bbox=(x, y, w, h),
                contour=cnt,
                area=area,
                aspect_ratio=round(aspect_ratio, 2),
                extent=round(extent, 3),
                solidity=round(solidity, 3),
                mean_intensity_drop=round(intensity_drop, 2),
                texture_features=texture_feats
            )
            scored_defects.append(defect)
            cv2.drawContours(fused_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        # Compute heatmap
        heatmap = self.compute_anomaly_heatmap(enhanced_image, fused_mask)

        # Overall anomaly score: ratio of defect pixels weighted by confidence
        total_pixels = enhanced_image.shape[0] * enhanced_image.shape[1]
        defect_pixels = np.count_nonzero(fused_mask)
        score = float(min(1.0, (defect_pixels / total_pixels) * 50.0))

        return AnomalyResult(
            fused_binary_mask=fused_mask,
            anomaly_heatmap=heatmap,
            detected_defects=scored_defects,
            overall_anomaly_score=round(score, 4)
        )
