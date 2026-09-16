"""
VisionInspect Command-Line Interface (CLI).
Provides terminal commands for single inspection, batch processing, benchmarking, and synthetic data generation.
"""

import sys
import argparse
from pathlib import Path
import json

from vision_inspect import __version__
from vision_inspect.config import InspectionConfig
from vision_inspect.pipeline import InspectionPipeline
from vision_inspect.benchmark import BenchmarkEngine


def print_banner():
    banner = r"""
========================================================================
     _   _ _     _             ___                     _   
    | | | (_)___(_)___  _ __  |_ _|_ __  ___ _ __  ___| |_ 
    | | | | / __| / _ \| '_ \  | || '_ \/ __| '_ \/ _ \ __|
    | |_| | \__ \ | (_) | | | | | || | | \__ \ |_) |  __/ |_ 
     \___/|_|___/_|\___/|_| |_|___|_| |_|___/ .__/ \___|\__|
                                            |_|             
    Industrial Surface Defect Inspection System | v""" + f"{__version__}\n" + r"""========================================================================
"""
    print(banner)


def run_inspect(args) -> int:
    """Handles single image inspection."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[-] Error: Input image '{args.input}' does not exist.")
        return 2

    config = InspectionConfig()
    if args.sensitivity == "high":
        config.classical.gradient_threshold = 25
        config.classical.min_defect_area = 8
    elif args.sensitivity == "low":
        config.classical.gradient_threshold = 45
        config.classical.min_defect_area = 25

    if args.max_defects is not None:
        config.max_tolerated_defects = args.max_defects

    pipeline = InspectionPipeline(config)
    output_dir = Path(args.output_dir) if args.output_dir else Path("output")
    
    print(f"[*] Ingesting: {input_path.name}")
    execution = pipeline.inspect(input_path, output_dir=output_dir)
    rep = execution.report

    # Terminal summary output
    verdict_badge = "[PASS]" if rep.verdict == "PASS" else "[FAIL - REJECTED]"
    print("\n" + "=" * 55)
    print(f" INSPECTION RESULT: {verdict_badge}")
    print("=" * 55)
    print(f" - Sample ID         : {execution.sample_id}")
    print(f" - Latency           : {execution.runtime_ms:.2f} ms")
    print(f" - Total Defects     : {rep.total_defects}")
    print(f"   * Critical        : {rep.critical_defects}")
    print(f"   * Major           : {rep.major_defects}")
    print(f"   * Minor           : {rep.minor_defects}")
    print(f" - Defect Coverage   : {rep.defect_coverage_percent:.4f}% ({rep.total_defect_area_px} px)")
    
    if rep.rejection_reasons:
        print("\n [!] Rejection Findings:")
        for r in rep.rejection_reasons:
            print(f"     * {r}")

    if rep.defect_metrics:
        print("\n [Defect Inventory]")
        for d in rep.defect_metrics:
            print(f"  [{d.severity:8s}] Defect #{d.defect_id}: {d.defect_type.capitalize():10s} | Area: {d.pixel_area:6.1f}px | BBox: {d.bbox} | Conf: {d.confidence:.2f}")

    print("\n [Artifacts Saved]")
    for k, v in execution.output_files.items():
        print(f"  + {k.capitalize():16s} -> {v}")
    print("=" * 55 + "\n")

    return 0 if rep.verdict == "PASS" else 1


def run_batch(args) -> int:
    """Handles directory-wide batch inspection."""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path("output")

    if not input_dir.exists():
        print(f"[-] Error: Directory '{args.input_dir}' not found.")
        return 2

    config = InspectionConfig()
    pipeline = InspectionPipeline(config)

    print(f"[*] Running batch inspection on directory: {input_dir}")
    try:
        executions = pipeline.batch_inspect(input_dir, output_dir)
    except Exception as e:
        print(f"[-] Batch execution failed: {e}")
        return 2

    total = len(executions)
    passed = sum(1 for e in executions if e.report.verdict == "PASS")
    failed = total - passed
    pass_rate = (passed / total) * 100.0 if total > 0 else 0.0

    print("\n" + "=" * 55)
    print(" BATCH INSPECTION SUMMARY")
    print("=" * 55)
    print(f" - Total Inspected  : {total}")
    print(f" - Passed           : {passed}")
    print(f" - Rejected         : {failed}")
    print(f" - Line Pass Rate   : {pass_rate:.1f}%")
    print(f" - Summary CSV      : {output_dir / 'batch_inspection_summary.csv'}")
    print("=" * 55 + "\n")

    return 0


def run_benchmark(args) -> int:
    """Runs evaluation benchmark against ground-truth masks."""
    data_dir = Path(args.data_dir)
    gt_dir = Path(args.gt_dir)

    if not data_dir.exists() or not gt_dir.exists():
        print("[-] Error: Data directory or ground-truth directory not found.")
        return 2

    pipeline = InspectionPipeline()
    benchmarker = BenchmarkEngine(pipeline)

    print(f"[*] Evaluating pipeline against ground truth in: {gt_dir}")
    try:
        summary = benchmarker.evaluate(data_dir, gt_dir)
    except Exception as e:
        print(f"[-] Benchmark failed: {e}")
        return 2

    print("\n" + "=" * 55)
    print(" BENCHMARK ACCURACY & SEGMENTATION METRICS")
    print("=" * 55)
    print(f" - Evaluated Samples: {summary.total_samples}")
    print(f" - Mean Precision   : {summary.mean_precision * 100.0:.2f}%")
    print(f" - Mean Recall      : {summary.mean_recall * 100.0:.2f}%")
    print(f" - Mean F1-Score    : {summary.mean_f1_score * 100.0:.2f}%")
    print(f" - Mean IoU         : {summary.mean_iou * 100.0:.2f}%")
    print(f" - Avg Latency      : {summary.average_latency_ms:.2f} ms")
    print("=" * 55)

    if args.export_json:
        out_json = Path(args.export_json)
        data = {
            "total_samples": summary.total_samples,
            "mean_precision": summary.mean_precision,
            "mean_recall": summary.mean_recall,
            "mean_f1_score": summary.mean_f1_score,
            "mean_iou": summary.mean_iou,
            "average_latency_ms": summary.average_latency_ms,
            "samples": [
                {
                    "sample_id": s.sample_id,
                    "precision": s.precision,
                    "recall": s.recall,
                    "f1_score": s.f1_score,
                    "iou": s.iou,
                    "tp": s.true_positives,
                    "fp": s.false_positives,
                    "fn": s.false_negatives
                }
                for s in summary.sample_metrics
            ]
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f" [+] Exported benchmark metrics to: {out_json}\n")

    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="vision-inspect",
        description="VisionInspect: Industrial Surface Defect Detection & Quality Inspection Pipeline."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Inspection command mode")

    # Inspect command
    inspect_p = subparsers.add_parser("inspect", help="Inspect a single surface image")
    inspect_p.add_argument("--input", "-i", required=True, help="Path to input image file")
    inspect_p.add_argument("--output-dir", "-o", default="output", help="Directory to store inspection artifacts")
    inspect_p.add_argument("--sensitivity", choices=["low", "normal", "high"], default="normal", help="Detection sensitivity level")
    inspect_p.add_argument("--max-defects", type=int, default=None, help="Maximum allowed defects before flagging FAIL")

    # Batch command
    batch_p = subparsers.add_parser("batch", help="Inspect an entire directory of surface images")
    batch_p.add_argument("--input-dir", "-i", required=True, help="Input directory containing images")
    batch_p.add_argument("--output-dir", "-o", default="output", help="Output directory for reports and logs")

    # Benchmark command
    bench_p = subparsers.add_parser("benchmark", help="Evaluate segmentation metrics against ground truth masks")
    bench_p.add_argument("--data-dir", "-d", required=True, help="Directory containing test images")
    bench_p.add_argument("--gt-dir", "-g", required=True, help="Directory containing ground-truth masks")
    bench_p.add_argument("--export-json", default=None, help="Path to save benchmark results JSON")

    args = parser.parse_args()

    if not args.command:
        print_banner()
        parser.print_help()
        sys.exit(0)

    print_banner()

    if args.command == "inspect":
        sys.exit(run_inspect(args))
    elif args.command == "batch":
        sys.exit(run_batch(args))
    elif args.command == "benchmark":
        sys.exit(run_benchmark(args))


if __name__ == "__main__":
    main()
