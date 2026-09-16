"""
Unit tests for Spatial Analyzer and Quality Pass/Fail grading.
"""

import numpy as np
import pytest

from vision_inspect.config import InspectionConfig
from vision_inspect.anomaly_detector import ScoredDefect, AnomalyResult
from vision_inspect.feature_extractor import TextureFeatures
from vision_inspect.spatial_analyzer import SpatialAnalyzer, QualityInspectionReport


def create_mock_defect(defect_id=1, defect_type="pitting", area=50.0):
    dummy_feats = TextureFeatures(0, 0, 1, 1, 1, 100, 10, 2)
    return ScoredDefect(
        defect_id=defect_id,
        defect_type=defect_type,
        confidence=0.9,
        bbox=(20, 20, 10, 10),
        contour=np.array([[[20, 20]], [[30, 20]], [[30, 30]], [[20, 30]]]),
        area=area,
        aspect_ratio=1.0,
        extent=0.5,
        solidity=0.9,
        mean_intensity_drop=20.0,
        texture_features=dummy_feats
    )


def test_spatial_analyzer_pass_verdict():
    analyzer = SpatialAnalyzer()
    dummy_anomaly = AnomalyResult(
        fused_binary_mask=np.zeros((100, 100), dtype=np.uint8),
        anomaly_heatmap=np.zeros((100, 100), dtype=np.uint8),
        detected_defects=[],
        overall_anomaly_score=0.0
    )
    report = analyzer.analyze(dummy_anomaly, (100, 100))
    assert isinstance(report, QualityInspectionReport)
    assert report.verdict == "PASS"
    assert report.total_defects == 0


def test_spatial_analyzer_fail_verdict_on_crack():
    analyzer = SpatialAnalyzer()
    crack_defect = create_mock_defect(defect_type="crack", area=200.0)
    anomaly = AnomalyResult(
        fused_binary_mask=np.zeros((100, 100), dtype=np.uint8),
        anomaly_heatmap=np.zeros((100, 100), dtype=np.uint8),
        detected_defects=[crack_defect],
        overall_anomaly_score=0.2
    )
    report = analyzer.analyze(anomaly, (100, 100))
    assert report.verdict == "FAIL"
    assert report.critical_defects == 1
    assert any("CRITICAL structural defect" in r for r in report.rejection_reasons)
