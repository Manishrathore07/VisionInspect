"""
Classical Morphological and Gradient-Based Defect Detection Engine.
Extracts edge contours, morphological gradients, and candidate defect regions.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import cv2
import numpy as np

from vision_inspect.config import ClassicalDetectionConfig


@dataclass
class ClassicalDetectionResult:
    """Output container for classical detection stage."""
    binary_mask: np.ndarray
    gradient_magnitude: np.ndarray
    adaptive_mask: np.ndarray
    candidate_contours: List[np.ndarray]
    candidate_boxes: List[Tuple[int, int, int, int]]  # (x, y, w, h)


class MorphologicalEngine:
    """
    Detects high-frequency anomalies, cracks, scratches, and pitting using
    differential operators, adaptive binarization, and mathematical morphology.
    """

    def __init__(self, config: Optional[ClassicalDetectionConfig] = None):
        self.config = config or ClassicalDetectionConfig()

    def compute_gradient(self, image: np.ndarray) -> np.ndarray:
        """
        Computes the spatial gradient magnitude using Sobel operators.
        """
        ksize = self.config.gradient_ksize
        grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=ksize)
        
        magnitude = cv2.magnitude(grad_x, grad_y)
        # Normalize to 8-bit unsigned integer
        magnitude_8u = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return magnitude_8u

    def adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """
        Applies local adaptive Gaussian thresholding to isolate local deviations
        relative to the immediate neighborhood mean.
        """
        bsize = self.config.adaptive_block_size
        if bsize % 2 == 0:
            bsize += 1
        return cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            bsize,
            self.config.adaptive_c
        )

    def morphological_refinement(self, mask: np.ndarray) -> np.ndarray:
        """
        Applies morphological opening (noise suppression) followed by closing
        (bridging fragmented hairline defects).
        """
        k = self.config.morph_kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        
        # Opening removes isolated single-pixel salt noise
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        # Closing bridges small gaps along cracks
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=self.config.morph_iterations)
        return closed

    def detect(self, enhanced_image: np.ndarray) -> ClassicalDetectionResult:
        """
        Executes classical defect isolation on the enhanced image.
        """
        # Step 1: Gradient computation
        gradient_map = self.compute_gradient(enhanced_image)
        _, grad_binary = cv2.threshold(
            gradient_map,
            self.config.gradient_threshold,
            255,
            cv2.THRESH_BINARY
        )

        # Step 2: Adaptive neighborhood thresholding
        adapt_binary = self.adaptive_threshold(enhanced_image)

        # Step 3: Multi-cue fusion (combines strong gradient boundaries and local intensity drops)
        combined_mask = cv2.bitwise_or(grad_binary, adapt_binary)

        # Suppress outermost image border to prevent edge-padding convolution artifacts
        border_margin = 8
        h, w = combined_mask.shape[:2]
        combined_mask[:border_margin, :] = 0
        combined_mask[-border_margin:, :] = 0
        combined_mask[:, :border_margin] = 0
        combined_mask[:, -border_margin:] = 0

        # Step 4: Morphological cleanup
        refined_mask = self.morphological_refinement(combined_mask)

        # Step 5: Contour extraction & geometric filtering
        contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours: List[np.ndarray] = []
        valid_boxes: List[Tuple[int, int, int, int]] = []
        filtered_mask = np.zeros_like(refined_mask)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.config.min_defect_area <= area <= self.config.max_defect_area:
                x, y, bw, bh = cv2.boundingRect(cnt)
                # Ignore artifacts glued to the boundary edge
                if x <= border_margin or y <= border_margin or (x + bw) >= (w - border_margin) or (y + bh) >= (h - border_margin):
                    continue
                valid_contours.append(cnt)
                valid_boxes.append((x, y, bw, bh))
                cv2.drawContours(filtered_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        return ClassicalDetectionResult(
            binary_mask=filtered_mask,
            gradient_magnitude=gradient_map,
            adaptive_mask=adapt_binary,
            candidate_contours=valid_contours,
            candidate_boxes=valid_boxes
        )
