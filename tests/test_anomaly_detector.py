"""
Unit tests for Anomaly Detection and Defect Classification.
"""

import cv2
import numpy as np
import pytest

from vision_inspect.config import InspectionConfig
from vision_inspect.classical_engine import MorphologicalEngine
from vision_inspect.anomaly_detector import AnomalyDetector, AnomalyResult


def test_classify_crack_shape():
    detector = AnomalyDetector()
    # Artificial long crack contour
    pts = np.array([[10, 10], [12, 10], [12, 120], [10, 120]], dtype=np.int32)
    patch = np.full((110, 2), 50, dtype=np.uint8)
    dtype, conf = detector.classify_defect_geometry(pts, patch, (10, 10, 2, 110))
    
    assert dtype in ["crack", "scratch"]
    assert conf > 0.70


def test_classify_pitting_shape():
    detector = AnomalyDetector()
    # Circular contour
    circle_img = np.zeros((50, 50), dtype=np.uint8)
    cv2.circle(circle_img, (25, 25), 10, 255, -1)
    contours, _ = cv2.findContours(circle_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    
    patch = np.zeros((20, 20), dtype=np.uint8)
    dtype, conf = detector.classify_defect_geometry(cnt, patch, (15, 15, 20, 20))
    assert dtype == "pitting"
    assert conf > 0.75
