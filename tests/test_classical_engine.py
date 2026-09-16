"""
Unit tests for the Classical Morphological and Gradient Engine.
"""

import cv2
import numpy as np
import pytest

from vision_inspect.config import ClassicalDetectionConfig
from vision_inspect.classical_engine import MorphologicalEngine, ClassicalDetectionResult


@pytest.fixture
def clean_image():
    """Uniform gray field with zero defects."""
    return np.full((200, 200), 128, dtype=np.uint8)


@pytest.fixture
def defective_image():
    """Image with an artificial high-contrast line defect in the center."""
    img = np.full((200, 200), 180, dtype=np.uint8)
    # Draw dark vertical defect (crack-like)
    cv2.line(img, (100, 40), (100, 160), color=20, thickness=3)
    return img


def test_classical_engine_clean_surface(clean_image):
    engine = MorphologicalEngine()
    result = engine.detect(clean_image)
    assert isinstance(result, ClassicalDetectionResult)
    assert len(result.candidate_contours) == 0
    assert np.count_nonzero(result.binary_mask) == 0


def test_classical_engine_detects_defect(defective_image):
    engine = MorphologicalEngine()
    result = engine.detect(defective_image)
    assert len(result.candidate_contours) >= 1
    assert np.count_nonzero(result.binary_mask) > 0


def test_gradient_magnitude(defective_image):
    engine = MorphologicalEngine()
    grad = engine.compute_gradient(defective_image)
    assert grad.shape == defective_image.shape
    assert grad.dtype == np.uint8
    # Center where line exists should have non-zero gradient
    assert np.max(grad[40:160, 95:105]) > 50
