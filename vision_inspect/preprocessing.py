"""
Image Preprocessing and Radiometric Correction Module.
Implements CLAHE, bilateral edge-preserving smoothing, and uneven illumination normalization.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union, Tuple, Optional
import cv2
import numpy as np

from vision_inspect.config import PreprocessingConfig


@dataclass
class PreprocessedData:
    """Container for processed intermediate image representations."""
    original_bgr: np.ndarray
    grayscale: np.ndarray
    illumination_corrected: np.ndarray
    denoised: np.ndarray
    enhanced: np.ndarray
    scale_factor: Tuple[float, float]  # (scale_x, scale_y) relative to raw input


class Preprocessor:
    """
    Handles image ingestion, radiometric normalization, noise reduction, and contrast enhancement.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_grid_size
        )

    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Loads an image from filesystem, verifying integrity and color channels.
        """
        path_str = str(image_path)
        img = cv2.imread(path_str, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image at: {path_str}")
        return img

    def normalize_illumination(self, gray: np.ndarray) -> np.ndarray:
        """
        Estimates non-uniform background illumination via morphological opening
        and removes low-frequency lighting gradients.
        """
        ksize = self.config.illumination_kernel_size
        # Kernel must be odd
        if ksize % 2 == 0:
            ksize += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        
        # Estimate background illumination
        background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # Subtract background and normalize to 0-255 range
        diff = cv2.subtract(gray, background)
        normalized = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        return normalized

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Applies bilateral filtering to eliminate high-frequency sensor noise
        while preserving crisp defect boundaries/edges.
        """
        return cv2.bilateralFilter(
            image,
            d=self.config.bilateral_d,
            sigmaColor=self.config.bilateral_sigma_color,
            sigmaSpace=self.config.bilateral_sigma_space
        )

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
        """
        return self.clahe.apply(image)

    def process(self, image_input: Union[str, Path, np.ndarray]) -> PreprocessedData:
        """
        Executes the full preprocessing pipeline on an image path or in-memory numpy array.
        """
        if isinstance(image_input, (str, Path)):
            raw_bgr = self.load_image(image_input)
        elif isinstance(image_input, np.ndarray):
            raw_bgr = image_input.copy()
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        orig_h, orig_w = raw_bgr.shape[:2]

        # Resize if target_size is specified
        if self.config.target_size:
            target_w, target_h = self.config.target_size
            resized_bgr = cv2.resize(raw_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
            scale_x = target_w / float(orig_w)
            scale_y = target_h / float(orig_h)
        else:
            resized_bgr = raw_bgr
            scale_x = 1.0
            scale_y = 1.0

        # Convert to Grayscale
        if len(resized_bgr.shape) == 3 and resized_bgr.shape[2] == 3:
            gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized_bgr.copy()

        # Step 1: Illumination normalization
        illum_corrected = self.normalize_illumination(gray)

        # Step 2: Edge-preserving smoothing
        denoised = self.denoise(illum_corrected)

        # Step 3: Local contrast equalization
        enhanced = self.enhance_contrast(denoised)

        return PreprocessedData(
            original_bgr=resized_bgr,
            grayscale=gray,
            illumination_corrected=illum_corrected,
            denoised=denoised,
            enhanced=enhanced,
            scale_factor=(scale_x, scale_y)
        )
