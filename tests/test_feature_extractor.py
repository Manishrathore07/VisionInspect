"""
Unit tests for Texture Feature Extraction and GLCM statistics.
"""

import numpy as np
import pytest

from vision_inspect.config import TextureFeatureConfig
from vision_inspect.feature_extractor import TextureFeatureExtractor, TextureFeatures


def test_texture_extractor_flat_patch():
    extractor = TextureFeatureExtractor()
    flat = np.full((32, 32), 100, dtype=np.uint8)
    feats = extractor.compute_patch_features(flat)

    assert isinstance(feats, TextureFeatures)
    assert feats.contrast == 0.0
    assert feats.homogeneity == pytest.approx(1.0, abs=0.01)
    assert feats.std_intensity == 0.0


def test_texture_extractor_high_variance_patch():
    extractor = TextureFeatureExtractor()
    # Alternating high contrast stripes
    noisy = np.zeros((32, 32), dtype=np.uint8)
    noisy[:, ::2] = 255
    feats = extractor.compute_patch_features(noisy)

    assert feats.contrast > 0.0
    assert feats.std_intensity > 50.0


def test_texture_grid_extraction():
    extractor = TextureFeatureExtractor(TextureFeatureConfig(patch_size=32, stride=16))
    img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    grid = extractor.extract_patch_grid(img)
    # Expected grid: ((64-32)//16 + 1) = 3 x 3
    assert grid.shape == (3, 3)
