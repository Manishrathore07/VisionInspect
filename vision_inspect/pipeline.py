"""
Master Pipeline Coordinator for VisionInspect.
Chains preprocessing, classical morphological engine, anomaly detection, spatial metrology, visualization, and audit export.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, List, Dict, Any
import time
import cv2
import numpy as np

from vision_inspect.config import InspectionConfig
from vision_inspect.preprocessing import Preprocessor, PreprocessedData
from vision_inspect.classical_engine import MorphologicalEngine, ClassicalDetectionResult
from vision_inspect.anomaly_detector import AnomalyDetector, AnomalyResult
from vision_inspect.spatial_analyzer import SpatialAnalyzer, QualityInspectionReport
from vision_inspect.visualizer import Visualizer
from vision_inspect.reporter import InspectionReporter


@dataclass
class InspectionExecution:
    """Full execution payload containing intermediate and final inspection results."""
    sample_id: str
    runtime_ms: float
    preprocessed: PreprocessedData
    classical_result: ClassicalDetectionResult
    anomaly_result: AnomalyResult
    report: QualityInspectionReport
    output_files: Dict[str, Path]


class InspectionPipeline:
    """
    End-to-end industrial surface inspection workflow coordinator.
    """

    def __init__(self, config: Optional[InspectionConfig] = None):
        self.config = config or InspectionConfig()
        self.preprocessor = Preprocessor(self.config.preprocessing)
        self.classical_engine = MorphologicalEngine(self.config.classical)
        self.anomaly_detector = AnomalyDetector(self.config)
        self.spatial_analyzer = SpatialAnalyzer(self.config)
        self.visualizer = Visualizer()

    def inspect(
        self,
        image_input: Union[str, Path, np.ndarray],
        sample_id: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None
    ) -> InspectionExecution:
        """
        Executes complete quality inspection on a single surface image.
        """
        start_time = time.perf_counter()

        # Determine sample ID
        if sample_id is None:
            if isinstance(image_input, (str, Path)):
                sample_id = Path(image_input).stem
            else:
                sample_id = f"sample_{int(time.time())}"

        # 1. Preprocessing & Radiometric correction
        prep_data = self.preprocessor.process(image_input)

        # 2. Classical Gradient & Morphological Candidate Extraction
        classical_result = self.classical_engine.detect(prep_data.enhanced)

        # 3. Anomaly Scoring & Geometric Defect Classification
        anomaly_result = self.anomaly_detector.detect(prep_data.enhanced, classical_result)

        # 4. Spatial Metrology & Quality Decision Evaluation
        report = self.spatial_analyzer.analyze(anomaly_result, prep_data.original_bgr.shape)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # 5. Rendering Visuals and Writing Reports
        output_files: Dict[str, Path] = {}

        if output_dir:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            reporter = InspectionReporter(out_p)

            # JSON Record
            if self.config.generate_json:
                json_path = reporter.save_json(sample_id, report, elapsed_ms)
                output_files["json"] = json_path

            # Markdown Certificate
            cert_content = reporter.generate_markdown_certificate(sample_id, report, elapsed_ms)
            output_files["certificate"] = out_p / f"{sample_id}_certificate.md"

            # Visuals
            if self.config.save_visualizations:
                overlay_img = self.visualizer.render_overlay(prep_data.original_bgr, anomaly_result, report)
                overlay_path = out_p / f"{sample_id}_overlay.png"
                self.visualizer.save(overlay_img, overlay_path)
                output_files["overlay"] = overlay_path

                composite_img = self.visualizer.render_composite(prep_data, anomaly_result, report)
                composite_path = out_p / f"{sample_id}_diagnostic_card.png"
                self.visualizer.save(composite_img, composite_path)
                output_files["diagnostic_card"] = composite_path

            if self.config.save_heatmaps:
                heatmap_img = self.visualizer.render_thermal_heatmap(anomaly_result.anomaly_heatmap)
                heatmap_path = out_p / f"{sample_id}_heatmap.png"
                self.visualizer.save(heatmap_img, heatmap_path)
                output_files["heatmap"] = heatmap_path

        return InspectionExecution(
            sample_id=sample_id,
            runtime_ms=round(elapsed_ms, 2),
            preprocessed=prep_data,
            classical_result=classical_result,
            anomaly_result=anomaly_result,
            report=report,
            output_files=output_files
        )

    def batch_inspect(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        extensions: tuple = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    ) -> List[InspectionExecution]:
        """
        Executes sequential inspection across an entire directory of surface images.
        Generates aggregate CSV audit logs.
        """
        in_p = Path(input_dir)
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        reporter = InspectionReporter(out_p)

        image_files = [f for f in sorted(in_p.glob("*")) if f.suffix.lower() in extensions]
        if not image_files:
            raise FileNotFoundError(f"No valid image files found in: {input_dir}")

        executions: List[InspectionExecution] = []
        batch_records: List[Dict[str, Any]] = []

        for img_path in image_files:
            execution = self.inspect(img_path, output_dir=out_p)
            executions.append(execution)
            batch_records.append(reporter.to_dict(execution.sample_id, execution.report, execution.runtime_ms))

        # Write Batch CSV
        reporter.save_batch_csv(batch_records, "batch_inspection_summary.csv")
        return executions
