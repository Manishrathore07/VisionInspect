"""
Spatial Metrology and Quality Severity Grading Module.
Measures defect dimensions, pixel areas, equivalent diameters, and assigns industrial quality grades.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

from vision_inspect.config import SeverityConfig, InspectionConfig
from vision_inspect.anomaly_detector import ScoredDefect, AnomalyResult


@dataclass
class DefectMetric:
    """Quantitative metrological measurement and severity grade for a single defect."""
    defect_id: int
    defect_type: str
    severity: str  # 'CRITICAL', 'MAJOR', 'MINOR'
    confidence: float
    bbox: List[int]  # [x, y, w, h]
    pixel_area: float
    area_ratio_percent: float
    equivalent_diameter: float
    aspect_ratio: float
    center_point: List[int]  # [center_x, center_y]


@dataclass
class QualityInspectionReport:
    """Overall quality decision and summary statistics for an inspected part."""
    verdict: str  # 'PASS' or 'FAIL'
    total_defects: int
    critical_defects: int
    major_defects: int
    minor_defects: int
    total_defect_area_px: float
    total_surface_area_px: int
    defect_coverage_percent: float
    defect_metrics: List[DefectMetric]
    rejection_reasons: List[str]


class SpatialAnalyzer:
    """
    Computes spatial dimensions, equivalent diameters, and evaluates parts
    against industry tolerance quality standards.
    """

    def __init__(self, config: Optional[InspectionConfig] = None):
        self.config = config or InspectionConfig()
        self.sev_cfg = self.config.severity

    def grade_severity(self, defect: ScoredDefect) -> str:
        """
        Assigns severity grade ('CRITICAL', 'MAJOR', 'MINOR') based on area and morphology.
        """
        # Linear structural failures (cracks) are always critical in manufacturing
        if defect.defect_type in self.sev_cfg.critical_defect_types and defect.area > self.sev_cfg.minor_max_area:
            return "CRITICAL"

        if defect.area > self.sev_cfg.major_max_area:
            return "CRITICAL"
        elif defect.area > self.sev_cfg.minor_max_area:
            return "MAJOR"
        else:
            return "MINOR"

    def analyze(self, anomaly_result: AnomalyResult, image_shape: tuple) -> QualityInspectionReport:
        """
        Executes comprehensive metrology on all detected defects and produces pass/fail verdict.
        """
        h, w = image_shape[:2]
        total_surface_area = h * w
        total_defect_area = 0.0

        critical_cnt = 0
        major_cnt = 0
        minor_cnt = 0
        metrics: List[DefectMetric] = []
        rejection_reasons: List[str] = []

        for defect in anomaly_result.detected_defects:
            total_defect_area += defect.area
            severity = self.grade_severity(defect)

            if severity == "CRITICAL":
                critical_cnt += 1
            elif severity == "MAJOR":
                major_cnt += 1
            else:
                minor_cnt += 1

            x, y, bw, bh = defect.bbox
            center_x = x + bw // 2
            center_y = y + bh // 2

            # Equivalent circular diameter = sqrt(4 * area / pi)
            eq_diameter = round(float(np.sqrt(4.0 * defect.area / np.pi)), 2)
            area_ratio = round((defect.area / total_surface_area) * 100.0, 4)

            metric = DefectMetric(
                defect_id=defect.defect_id,
                defect_type=defect.defect_type,
                severity=severity,
                confidence=defect.confidence,
                bbox=[x, y, bw, bh],
                pixel_area=round(defect.area, 1),
                area_ratio_percent=area_ratio,
                equivalent_diameter=eq_diameter,
                aspect_ratio=defect.aspect_ratio,
                center_point=[center_x, center_y]
            )
            metrics.append(metric)

        coverage_percent = round((total_defect_area / total_surface_area) * 100.0, 4)

        # Quality Verdict Evaluation
        verdict = "PASS"
        if len(anomaly_result.detected_defects) > self.config.max_tolerated_defects:
            verdict = "FAIL"
            rejection_reasons.append(
                f"Exceeded max defect tolerance: found {len(anomaly_result.detected_defects)}, allowed {self.config.max_tolerated_defects}"
            )
        if total_defect_area > self.config.max_tolerated_defect_area:
            if verdict == "PASS":
                verdict = "FAIL"
            rejection_reasons.append(
                f"Exceeded total defect area limit: found {total_defect_area} px, allowed {self.config.max_tolerated_defect_area} px"
            )
        if critical_cnt > 0:
            verdict = "FAIL"
            rejection_reasons.append(f"Detected {critical_cnt} CRITICAL structural defect(s).")

        return QualityInspectionReport(
            verdict=verdict,
            total_defects=len(metrics),
            critical_defects=critical_cnt,
            major_defects=major_cnt,
            minor_defects=minor_cnt,
            total_defect_area_px=round(total_defect_area, 1),
            total_surface_area_px=total_surface_area,
            defect_coverage_percent=coverage_percent,
            defect_metrics=metrics,
            rejection_reasons=rejection_reasons
        )
