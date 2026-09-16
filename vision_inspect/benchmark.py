"""
Benchmark and Evaluation Engine.
Evaluates detection and segmentation performance against ground-truth defect masks.
Computes Precision, Recall, F1-Score, Specificity, and IoU (Intersection over Union).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union, List, Dict, Any, Tuple
import json
import cv2
import numpy as np

from vision_inspect.pipeline import InspectionPipeline


@dataclass
class SampleBenchmarkMetric:
    """Quantitative performance metrics for an individual test sample."""
    sample_id: str
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    iou: float


@dataclass
class BenchmarkSummary:
    """Aggregated benchmark statistics across an evaluation dataset."""
    total_samples: int
    mean_precision: float
    mean_recall: float
    mean_f1_score: float
    mean_iou: float
    average_latency_ms: float
    sample_metrics: List[SampleBenchmarkMetric]


class BenchmarkEngine:
    """
    Automated evaluation runner comparing predicted defect masks with ground-truth binary masks.
    """

    def __init__(self, pipeline: InspectionPipeline):
        self.pipeline = pipeline

    def compute_metrics(
        self,
        predicted_mask: np.ndarray,
        ground_truth_mask: np.ndarray,
        sample_id: str
    ) -> SampleBenchmarkMetric:
        """
        Computes pixel-level confusion matrix and derived statistical metrics.
        """
        # Ensure binary 0/1
        pred_bin = (predicted_mask > 0).astype(np.uint8)
        gt_bin = (ground_truth_mask > 0).astype(np.uint8)

        # Resize gt to pred if necessary
        if pred_bin.shape != gt_bin.shape:
            gt_bin = cv2.resize(gt_bin, (pred_bin.shape[1], pred_bin.shape[0]), interpolation=cv2.INTER_NEAREST)

        tp = int(np.sum((pred_bin == 1) & (gt_bin == 1)))
        fp = int(np.sum((pred_bin == 1) & (gt_bin == 0)))
        tn = int(np.sum((pred_bin == 0) & (gt_bin == 0)))
        fn = int(np.sum((pred_bin == 0) & (gt_bin == 1)))

        # Precision
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)

        # Recall
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)

        # F1 Score
        if (precision + recall) > 0:
            f1 = float(2.0 * (precision * recall) / (precision + recall))
        else:
            f1 = 0.0

        # Intersection over Union (IoU)
        intersection = tp
        union = tp + fp + fn
        iou = float(intersection / union) if union > 0 else 1.0

        return SampleBenchmarkMetric(
            sample_id=sample_id,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1, 4),
            iou=round(iou, 4)
        )

    def evaluate(
        self,
        images_dir: Union[str, Path],
        ground_truth_dir: Union[str, Path]
    ) -> BenchmarkSummary:
        """
        Evaluates the pipeline on a paired dataset of surface images and ground-truth masks.
        """
        img_p = Path(images_dir)
        gt_p = Path(ground_truth_dir)

        image_files = sorted([f for f in img_p.glob("*") if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]])
        if not image_files:
            raise FileNotFoundError(f"No sample images found in: {images_dir}")

        sample_metrics: List[SampleBenchmarkMetric] = []
        latencies: List[float] = []

        for img_file in image_files:
            stem = img_file.stem
            # Find matching ground truth (supports stem.png, stem_mask.png, stem_gt.png)
            gt_candidates = [
                gt_p / f"{stem}.png",
                gt_p / f"{stem}_mask.png",
                gt_p / f"{stem}_gt.png"
            ]
            gt_file = next((f for f in gt_candidates if f.exists()), None)
            if not gt_file:
                continue

            gt_mask = cv2.imread(str(gt_file), cv2.IMREAD_GRAYSCALE)
            if gt_mask is None:
                continue

            execution = self.pipeline.inspect(img_file)
            latencies.append(execution.runtime_ms)

            pred_mask = execution.anomaly_result.fused_binary_mask
            metric = self.compute_metrics(pred_mask, gt_mask, stem)
            sample_metrics.append(metric)

        if not sample_metrics:
            raise ValueError("No matching ground-truth masks found for evaluation.")

        mean_prec = float(np.mean([m.precision for m in sample_metrics]))
        mean_rec = float(np.mean([m.recall for m in sample_metrics]))
        mean_f1 = float(np.mean([m.f1_score for m in sample_metrics]))
        mean_iou = float(np.mean([m.iou for m in sample_metrics]))
        avg_lat = float(np.mean(latencies))

        return BenchmarkSummary(
            total_samples=len(sample_metrics),
            mean_precision=round(mean_prec, 4),
            mean_recall=round(mean_rec, 4),
            mean_f1_score=round(mean_f1, 4),
            mean_iou=round(mean_iou, 4),
            average_latency_ms=round(avg_lat, 2),
            sample_metrics=sample_metrics
        )
