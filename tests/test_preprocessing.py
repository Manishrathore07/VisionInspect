"""
Unit tests for the Preprocessor and radiometric normalization module.
"""

import pytest
import numpy as np

from vision_inspect.config import PreprocessingConfig
from vision_inspect.preprocessing import Preprocessor, PreprocessedData


@pytest.fixture
def synthetic_image():
    """Generates a simple 100x100 3-channel test image with an intensity gradient."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    for y in range(100):
        img[y, :, :] = int(y * 2.5)
    return img


def test_preprocessor_initialization():
    cfg = PreprocessingConfig(clahe_clip_limit=3.0)
    prep = Preprocessor(cfg)
    assert prep.config.clahe_clip_limit == 3.0


def test_illumination_normalization(synthetic_image):
    prep = Preprocessor()
    gray = synthetic_image[:, :, 0]
    norm = prep.normalize_illumination(gray)
    assert norm.shape == (100, 100)
    assert norm.dtype == np.uint8


def test_denoise_preserves_dimensions(synthetic_image):
    prep = Preprocessor()
    gray = synthetic_image[:, :, 0]
    denoised = prep.denoise(gray)
    assert denoised.shape == gray.shape


def test_full_preprocessing_pipeline(synthetic_image):
    cfg = PreprocessingConfig(target_size=(120, 120))
    prep = Preprocessor(cfg)
    res = prep.process(synthetic_image)

    assert isinstance(res, PreprocessedData)
    assert res.original_bgr.shape == (120, 120, 3)
    assert res.enhanced.shape == (120, 120)
    assert res.scale_factor[0] == 1.2
