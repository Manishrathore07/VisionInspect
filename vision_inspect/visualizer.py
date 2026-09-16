"""
Visualization and Inspection Artifact Rendering Module.
Generates color-coded overlays, thermal heatmaps, and side-by-side comparison cards.
"""

from pathlib import Path
from typing import Union, Optional
import cv2
import numpy as np

from vision_inspect.spatial_analyzer import QualityInspectionReport
from vision_inspect.anomaly_detector import AnomalyResult
from vision_inspect.preprocessing import PreprocessedData


class Visualizer:
    """
    Renders annotated inspection images, thermal heatmaps, and composite cards.
    """

    SEVERITY_COLORS = {
        "CRITICAL": (0, 0, 255),    # Red in BGR
        "MAJOR": (0, 140, 255),     # Orange in BGR
        "MINOR": (0, 255, 255)      # Yellow in BGR
    }

    def render_overlay(
        self,
        base_bgr: np.ndarray,
        anomaly_result: AnomalyResult,
        report: QualityInspectionReport
    ) -> np.ndarray:
        """
        Draws highlighted defect contours, color-coded bounding boxes, and severity tags.
        """
        annotated = base_bgr.copy()
        overlay = base_bgr.copy()

        # Step 1: Draw translucent contour fills
        for defect in anomaly_result.detected_defects:
            metric = next((m for m in report.defect_metrics if m.defect_id == defect.defect_id), None)
            severity = metric.severity if metric else "MINOR"
            color = self.SEVERITY_COLORS.get(severity, (0, 255, 0))

            cv2.drawContours(overlay, [defect.contour], -1, color, thickness=cv2.FILLED)

        # Blend contours with alpha = 0.35
        cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)

        # Step 2: Draw crisp bounding boxes and badges
        for defect in anomaly_result.detected_defects:
            metric = next((m for m in report.defect_metrics if m.defect_id == defect.defect_id), None)
            severity = metric.severity if metric else "MINOR"
            color = self.SEVERITY_COLORS.get(severity, (0, 255, 0))

            x, y, w, h = defect.bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            label = f"#{defect.defect_id} {defect.defect_type.upper()} ({severity[0]})"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x, max(0, y - lh - 6)), (x + lw + 4, y), color, cv2.FILLED)
            cv2.putText(annotated, label, (x + 2, max(lh, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # Step 3: Draw top status banner
        banner_h = 45
        banner = np.zeros((banner_h, annotated.shape[1], 3), dtype=np.uint8)
        banner_color = (0, 160, 0) if report.verdict == "PASS" else (0, 0, 180)
        banner[:] = banner_color

        status_text = f"INSPECTION VERDICT: {report.verdict} | DEFECTS: {report.total_defects} (Crit: {report.critical_defects}, Maj: {report.major_defects}, Min: {report.minor_defects})"
        cv2.putText(banner, status_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        annotated = np.vstack([banner, annotated])
        return annotated

    def render_thermal_heatmap(self, heatmap_gray: np.ndarray) -> np.ndarray:
        """
        Applies a JET colormap to the anomaly probability map.
        """
        return cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)

    def render_composite(
        self,
        prep_data: PreprocessedData,
        anomaly_result: AnomalyResult,
        report: QualityInspectionReport
    ) -> np.ndarray:
        """
        Produces a 2x2 multi-panel diagnostic card:
        [Raw Input]         | [CLAHE & Denoised]
        [Thermal Heatmap]   | [Defect Overlay]
        """
        raw = prep_data.original_bgr
        h, w = raw.shape[:2]

        # Panel 1: Raw
        p1 = raw.copy()
        cv2.putText(p1, "1. Raw Surface Input", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Panel 2: Preprocessed & enhanced
        p2 = cv2.cvtColor(prep_data.enhanced, cv2.COLOR_GRAY2BGR)
        cv2.putText(p2, "2. Radiometric Normalization", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Panel 3: Heatmap
        thermal = self.render_thermal_heatmap(anomaly_result.anomaly_heatmap)
        thermal = cv2.resize(thermal, (w, h))
        cv2.putText(thermal, "3. Anomaly Probability Energy", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Panel 4: Overlay without extra banner for symmetry
        p4 = self.render_overlay(raw, anomaly_result, report)
        # remove banner height for 2x2 tile alignment
        p4 = cv2.resize(p4, (w, h))
        cv2.putText(p4, f"4. Decision Overlay ({report.verdict})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Stitch 2x2
        top_row = np.hstack([p1, p2])
        bottom_row = np.hstack([thermal, p4])
        composite = np.vstack([top_row, bottom_row])
        return composite

    def save(self, image: np.ndarray, output_path: Union[str, Path]) -> None:
        """Saves image to specified destination."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), image)
