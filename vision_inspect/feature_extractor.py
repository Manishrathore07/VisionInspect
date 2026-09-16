"""
Texture and Morphometric Feature Extraction Module.
Computes Gray-Level Co-occurrence Matrix (GLCM) second-order statistics and patch descriptors.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

try:
    from skimage.feature import graycomatrix, graycoprops
    SKIMAGE_GLCM_AVAILABLE = True
except ImportError:
    SKIMAGE_GLCM_AVAILABLE = False

from vision_inspect.config import TextureFeatureConfig


@dataclass
class TextureFeatures:
    """Statistical texture descriptor values."""
    contrast: float
    dissimilarity: float
    homogeneity: float
    energy: float
    correlation: float
    mean_intensity: float
    std_intensity: float
    entropy: float


class TextureFeatureExtractor:
    """
    Extracts second-order statistical texture properties using Gray-Level Co-occurrence Matrices.
    Differentiates between uniform surface finishes, rough abrasions, and structural defects.
    """

    def __init__(self, config: Optional[TextureFeatureConfig] = None):
        self.config = config or TextureFeatureConfig()

    def _fallback_glcm_props(self, patch: np.ndarray) -> Dict[str, float]:
        """
        Pure NumPy fallback for GLCM calculation when skimage is unavailable or initializing.
        Computes horizontal adjacency (dx=1, dy=0) co-occurrence.
        """
        # Quantize patch to 16 levels to speed up calculation
        q_patch = (patch / 16).astype(np.int32)
        h, w = q_patch.shape
        co_matrix = np.zeros((16, 16), dtype=np.float64)

        for y in range(h):
            for x in range(w - 1):
                i = q_patch[y, x]
                j = q_patch[y, x + 1]
                co_matrix[i, j] += 1

        total = np.sum(co_matrix)
        if total > 0:
            p = co_matrix / total
        else:
            p = np.zeros((16, 16))

        i_indices, j_indices = np.indices((16, 16))
        diff = np.abs(i_indices - j_indices)

        contrast = float(np.sum(p * (diff ** 2)))
        dissimilarity = float(np.sum(p * diff))
        homogeneity = float(np.sum(p / (1.0 + diff ** 2)))
        energy = float(np.sum(p ** 2))

        # Correlation
        mu_i = np.sum(i_indices * p)
        mu_j = np.sum(j_indices * p)
        sigma_i = np.sqrt(np.sum(((i_indices - mu_i) ** 2) * p))
        sigma_j = np.sqrt(np.sum(((j_indices - mu_j) ** 2) * p))
        if sigma_i * sigma_j > 1e-6:
            correlation = float(np.sum((i_indices - mu_i) * (j_indices - mu_j) * p) / (sigma_i * sigma_j))
        else:
            correlation = 0.0

        return {
            "contrast": contrast,
            "dissimilarity": dissimilarity,
            "homogeneity": homogeneity,
            "energy": energy,
            "correlation": correlation
        }

    def compute_patch_features(self, patch: np.ndarray) -> TextureFeatures:
        """
        Computes comprehensive texture descriptors for a 2D grayscale image patch.
        """
        if patch.size == 0:
            return TextureFeatures(0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0)

        # 1. First-order statistical metrics
        mean_val = float(np.mean(patch))
        std_val = float(np.std(patch))
        
        # Shannon entropy
        hist, _ = np.histogram(patch, bins=32, range=(0, 256), density=True)
        hist = hist[hist > 0]
        entropy_val = float(-np.sum(hist * np.log2(hist))) if len(hist) > 0 else 0.0

        # 2. Second-order GLCM metrics
        if SKIMAGE_GLCM_AVAILABLE:
            try:
                # Quantize patch to 32 levels to balance accuracy and efficiency
                q_patch = (patch / 8).astype(np.uint8)
                glcm = graycomatrix(
                    q_patch,
                    distances=self.config.glcm_distances,
                    angles=self.config.glcm_angles,
                    levels=32,
                    symmetric=True,
                    normed=True
                )
                contrast = float(np.mean(graycoprops(glcm, 'contrast')))
                dissimilarity = float(np.mean(graycoprops(glcm, 'dissimilarity')))
                homogeneity = float(np.mean(graycoprops(glcm, 'homogeneity')))
                energy = float(np.mean(graycoprops(glcm, 'energy')))
                correlation = float(np.mean(graycoprops(glcm, 'correlation')))
            except Exception:
                props = self._fallback_glcm_props(patch)
                contrast, dissimilarity = props["contrast"], props["dissimilarity"]
                homogeneity, energy = props["homogeneity"], props["energy"]
                correlation = props["correlation"]
        else:
            props = self._fallback_glcm_props(patch)
            contrast, dissimilarity = props["contrast"], props["dissimilarity"]
            homogeneity, energy = props["homogeneity"], props["energy"]
            correlation = props["correlation"]

        return TextureFeatures(
            contrast=contrast,
            dissimilarity=dissimilarity,
            homogeneity=homogeneity,
            energy=energy,
            correlation=correlation,
            mean_intensity=mean_val,
            std_intensity=std_val,
            entropy=entropy_val
        )

    def extract_patch_grid(self, image: np.ndarray) -> np.ndarray:
        """
        Sliding-window texture mapping across the entire image.
        Returns a 2D anomaly contrast grid.
        """
        h, w = image.shape
        psize = self.config.patch_size
        stride = self.config.stride
        
        grid_y = (h - psize) // stride + 1
        grid_x = (w - psize) // stride + 1
        contrast_map = np.zeros((grid_y, grid_x), dtype=np.float32)

        for i, y in enumerate(range(0, h - psize + 1, stride)):
            for j, x in enumerate(range(0, w - psize + 1, stride)):
                patch = image[y:y + psize, x:x + psize]
                feat = self.compute_patch_features(patch)
                contrast_map[i, j] = feat.contrast

        return contrast_map
