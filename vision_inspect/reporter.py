"""
Reporting and Audit Trail Generation Module.
Outputs structured JSON records, CSV tabular summaries, and human-readable Markdown inspection certificates.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Union
from datetime import datetime

from vision_inspect.spatial_analyzer import QualityInspectionReport, DefectMetric


class InspectionReporter:
    """
    Handles serialization of quality decisions and audit logging.
    """

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self, sample_id: str, report: QualityInspectionReport, latency_ms: float) -> Dict[str, Any]:
        """
        Converts report into a clean dictionary representation.
        """
        return {
            "sample_id": sample_id,
            "timestamp": datetime.now().isoformat(),
            "verdict": report.verdict,
            "latency_ms": round(latency_ms, 2),
            "summary": {
                "total_defects": report.total_defects,
                "critical_defects": report.critical_defects,
                "major_defects": report.major_defects,
                "minor_defects": report.minor_defects,
                "total_defect_area_px": report.total_defect_area_px,
                "surface_area_px": report.total_surface_area_px,
                "defect_coverage_percent": report.defect_coverage_percent
            },
            "rejection_reasons": report.rejection_reasons,
            "defects": [
                {
                    "defect_id": d.defect_id,
                    "type": d.defect_type,
                    "severity": d.severity,
                    "confidence": d.confidence,
                    "bbox": d.bbox,
                    "area_px": d.pixel_area,
                    "coverage_percent": d.area_ratio_percent,
                    "equivalent_diameter_px": d.equivalent_diameter,
                    "aspect_ratio": d.aspect_ratio,
                    "center": d.center_point
                }
                for d in report.defect_metrics
            ]
        }

    def save_json(self, sample_id: str, report: QualityInspectionReport, latency_ms: float) -> Path:
        """
        Writes a single-part inspection JSON file.
        """
        data = self.to_dict(sample_id, report, latency_ms)
        filepath = self.output_dir / f"{sample_id}_inspection.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        return filepath

    def save_batch_csv(self, batch_records: List[Dict[str, Any]], filename: str = "batch_summary.csv") -> Path:
        """
        Exports a batch summary CSV containing key line performance metrics.
        """
        filepath = self.output_dir / filename
        fieldnames = [
            "sample_id", "timestamp", "verdict", "latency_ms",
            "total_defects", "critical_defects", "major_defects", "minor_defects",
            "defect_coverage_percent", "rejection_reasons"
        ]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in batch_records:
                writer.writerow({
                    "sample_id": rec["sample_id"],
                    "timestamp": rec["timestamp"],
                    "verdict": rec["verdict"],
                    "latency_ms": rec["latency_ms"],
                    "total_defects": rec["summary"]["total_defects"],
                    "critical_defects": rec["summary"]["critical_defects"],
                    "major_defects": rec["summary"]["major_defects"],
                    "minor_defects": rec["summary"]["minor_defects"],
                    "defect_coverage_percent": rec["summary"]["defect_coverage_percent"],
                    "rejection_reasons": "; ".join(rec["rejection_reasons"]) if rec["rejection_reasons"] else "None"
                })

        return filepath

    def generate_markdown_certificate(self, sample_id: str, report: QualityInspectionReport, latency_ms: float) -> str:
        """
        Formats an official QA Inspection Certificate in Markdown.
        """
        status_badge = "✅ **PASS**" if report.verdict == "PASS" else "❌ **FAIL (REJECTED)**"
        lines = [
            f"# Surface Quality Inspection Certificate: {sample_id}",
            f"- **Date & Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"- **Inspection Verdict:** {status_badge}",
            f"- **Processing Runtime:** {latency_ms:.2f} ms",
            f"- **Defect Coverage:** {report.defect_coverage_percent:.4f}% ({report.total_defect_area_px} px)",
            "",
            "## Quality Defect Breakdown",
            f"- **Total Anomalies:** {report.total_defects}",
            f"- **Critical Severity:** {report.critical_defects}",
            f"- **Major Severity:** {report.major_defects}",
            f"- **Minor Severity:** {report.minor_defects}",
            ""
        ]

        if report.rejection_reasons:
            lines.append("### Rejection Findings:")
            for reason in report.rejection_reasons:
                lines.append(f"- ⚠️ {reason}")
            lines.append("")

        if report.defect_metrics:
            lines.append("### Defect Metric Register")
            lines.append("| ID | Classification | Severity | Confidence | Area (px) | Aspect Ratio | Bounding Box [x, y, w, h] |")
            lines.append("|---|---|---|---|---|---|---|")
            for d in report.defect_metrics:
                lines.append(
                    f"| #{d.defect_id} | `{d.defect_type}` | **{d.severity}** | {d.confidence:.2f} | {d.pixel_area} | {d.aspect_ratio} | {d.bbox} |"
                )
        else:
            lines.append("✨ *No surface deviations or structural anomalies detected. Clean surface finish.*")

        md_content = "\n".join(lines)
        md_file = self.output_dir / f"{sample_id}_certificate.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(md_content)

        return md_content
