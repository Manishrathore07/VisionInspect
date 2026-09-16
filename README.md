# VisionInspect 🔍
### Automated Industrial Surface Defect Detection & Quality Inspection System

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Testing](https://img.shields.io/badge/pytest-17%20passed-brightgreen.svg)](https://pytest.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Interface](https://img.shields.io/badge/interface-CLI%20Headless-orange.svg)](#cli-usage)

**VisionInspect** is an open-source, modular Computer Vision system designed for high-speed industrial quality control and surface defect inspection. It detects and categorizes structural cracks, machining scratches, cavitation pitting, and material blemishes on metallic and composite surfaces using a hybrid architecture of morphological gradients, adaptive binarization, and Gray-Level Co-occurrence Matrix (GLCM) second-order texture statistics.

Built specifically for headless edge-devices, automated inspection lines, and evaluation pipelines, **VisionInspect operates 100% via the Command-Line Interface (CLI)** with deterministic exit codes and zero interactive GUI dependencies.

---

## 🌟 Key Features

- **100% Headless CLI Execution**: Designed for batch scripts, CI pipelines, and industrial terminal environments.
- **Radiometric Normalization**: Compensates for non-uniform ambient illumination gradients using morphological background subtraction, bilateral smoothing, and Contrast Limited Adaptive Histogram Equalization (CLAHE).
- **Multi-Cue Classical Detection Engine**: Fuses Sobel gradient energy, adaptive Gaussian thresholding, and morphological opening/closing operators to isolate sub-millimeter defects.
- **GLCM Statistical Texture Analysis**: Extracts second-order spatial statistics (contrast, dissimilarity, homogeneity, energy, correlation) and Shannon entropy.
- **Geometric Defect Classification**: Types defects into `crack`, `scratch`, `pitting`, or `blemish` using oriented bounding boxes, circularity, extent, and convex hull solidity.
- **Industrial Metrology & Severity Grading**: Measures pixel area, equivalent circular diameter, and coverage percentage, assigning industrial grades (`CRITICAL`, `MAJOR`, `MINOR`) and `PASS`/`FAIL` verdicts.
- **Multi-Format Audit Generation**: Exports structured JSON records, tabular CSV summaries, human-readable Markdown inspection certificates, thermal heatmaps, and composite diagnostic cards.
- **Automated Ground-Truth Benchmarking**: Evaluates Precision, Recall, F1-Score, and Mean Intersection-over-Union (mIoU) with sub-50ms per megapixel inference on standard CPUs.
- **Procedural Dataset Generator**: Built-in generator synthesizing realistic textured specimens with controlled defect patterns and paired ground-truth masks.

---

## 🛠️ Technologies & Libraries

- **Language**: Python 3.12+
- **Computer Vision & Image Processing**: OpenCV (`opencv-python`), Scikit-Image (`scikit-image`)
- **Numerical & Scientific Computing**: NumPy, SciPy
- **Data Visualization**: Matplotlib
- **Testing & Quality Assurance**: PyTest

---

## 📁 Repository Structure

```
VisionInspect/
├── data/
│   ├── samples/               # Sample specimen images (clean, cracks, scratches, pitting)
│   └── ground_truth/          # Paired binary segmentation masks for benchmarking
├── vision_inspect/
│   ├── __init__.py            # Package root & public API exports
│   ├── config.py              # Configuration dataclasses (thresholds, kernels, tolerances)
│   ├── preprocessing.py       # Illumination correction, bilateral filtering, CLAHE
│   ├── classical_engine.py    # Sobel gradients, adaptive thresholding, morphology
│   ├── feature_extractor.py   # GLCM texture descriptors & entropy metrics
│   ├── anomaly_detector.py    # Multi-cue anomaly fusion & geometric defect typing
│   ├── spatial_analyzer.py    # Spatial metrology, severity grading, pass/fail decisions
│   ├── visualizer.py          # Thermal heatmaps, translucent overlays, composite cards
│   ├── reporter.py            # JSON, CSV, and Markdown audit trail exporters
│   ├── pipeline.py            # Master pipeline coordinator
│   ├── benchmark.py           # Segmentation accuracy & IoU evaluation engine
│   └── cli.py                 # Command-line interface entry point
├── tests/
│   ├── test_preprocessing.py   # Radiometric normalization tests
│   ├── test_classical_engine.py# Gradient & morphological operator tests
│   ├── test_feature_extractor.py# GLCM & texture statistics tests
│   ├── test_anomaly_detector.py # Geometric defect classification tests
│   ├── test_spatial_analyzer.py # Severity grading & tolerance decision tests
│   └── test_cli.py             # CLI commands integration tests
├── scripts/
│   ├── generate_synthetic_data.py # Procedural benchmark dataset generator
│   └── export_report_html.py      # Exports report.md to formatted HTML/PDF
├── statement.md               # Official problem statement, scope & target users
├── README.md                  # Installation, configuration, and execution guide
├── report.md                  # Comprehensive 15-section academic project report
├── requirements.txt           # Pinned production dependencies
└── .gitignore                 # Standard Python git exclusions
```

---

## 🚀 Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/Manishrathore07/VisionInspect.git
cd VisionInspect
```

### 2. Set Up Virtual Environment (Python 3.12 recommended)
On Windows:
```cmd
py -3.12 -m venv .venv
.venv\Scripts\activate
```
On Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. (Optional) Generate Synthetic Benchmark Dataset
If you want to regenerate fresh surface samples and paired ground-truth masks:
```bash
python scripts/generate_synthetic_data.py
```

---

## 💻 CLI Usage

The primary interface is `vision_inspect.cli`. Run `--help` to inspect available commands:
```bash
python -m vision_inspect.cli --help
```

### 1. Inspect a Single Specimen
Inspects a specimen image, evaluates pass/fail status, and generates output artifacts (JSON, Markdown certificate, overlays, heatmap):
```bash
python -m vision_inspect.cli inspect --input data/samples/sample_01_clean.png --output-dir output
```
```bash
python -m vision_inspect.cli inspect --input data/samples/sample_03_crack.png --output-dir output
```

**CLI Exit Codes**:
- `0`: Inspection **PASS** (zero defect tolerance violated)
- `1`: Inspection **FAIL** (part rejected due to critical/major defects or tolerance breach)
- `2`: Runtime / filesystem error

**Optional Flags**:
- `--sensitivity {low, normal, high}`: Adjusts detection thresholds (default: `normal`).
- `--max-defects <N>`: Overrides tolerated defect count before failing (default: `0`).

---

### 2. Directory Batch Inspection
Processes an entire folder of parts, generating individual audit records and an aggregated `batch_inspection_summary.csv`:
```bash
python -m vision_inspect.cli batch --input-dir data/samples --output-dir output
```

**Sample Batch Terminal Output**:
```
=======================================================
 BATCH INSPECTION SUMMARY
=======================================================
 - Total Inspected  : 8
 - Passed           : 4
 - Rejected         : 4
 - Line Pass Rate   : 50.0%
 - Summary CSV      : output/batch_inspection_summary.csv
=======================================================
```

---

### 3. Evaluate Ground-Truth Segmentation Benchmark
Computes pixel-level Precision, Recall, F1-Score, and Mean Intersection-over-Union (mIoU) against ground-truth masks:
```bash
python -m vision_inspect.cli benchmark --data-dir data/samples --gt-dir data/ground_truth --export-json output/benchmark_results.json
```

**Sample Benchmark Output**:
```
=======================================================
 BENCHMARK ACCURACY & SEGMENTATION METRICS
=======================================================
 - Evaluated Samples: 8
 - Mean Precision   : 47.61%
 - Mean Recall      : 56.87%
 - Mean F1-Score    : 48.86%
 - Mean IoU         : 41.95%
 - Avg Latency      : 47.69 ms
=======================================================
 [+] Exported benchmark metrics to: output/benchmark_results.json
```

---

## 🧪 Automated Testing

VisionInspect includes a comprehensive test suite covering all modules, edge cases, and CLI workflows.
Execute all tests using `pytest`:
```bash
pytest tests/ -v
```

Expected output:
```
tests/test_anomaly_detector.py::test_classify_crack_shape PASSED         [  5%]
tests/test_anomaly_detector.py::test_classify_pitting_shape PASSED       [ 11%]
tests/test_classical_engine.py::test_classical_engine_clean_surface PASSED [ 17%]
tests/test_classical_engine.py::test_classical_engine_detects_defect PASSED [ 23%]
tests/test_classical_engine.py::test_gradient_magnitude PASSED           [ 29%]
tests/test_cli.py::test_cli_help PASSED                                  [ 35%]
tests/test_cli.py::test_cli_inspect_clean PASSED                         [ 41%]
tests/test_cli.py::test_cli_inspect_crack PASSED                         [ 47%]
tests/test_feature_extractor.py::test_texture_extractor_flat_patch PASSED [ 52%]
tests/test_feature_extractor.py::test_texture_extractor_high_variance_patch PASSED [ 58%]
tests/test_feature_extractor.py::test_texture_grid_extraction PASSED     [ 64%]
tests/test_preprocessing.py::test_preprocessor_initialization PASSED     [ 70%]
tests/test_preprocessing.py::test_illumination_normalization PASSED      [ 76%]
tests/test_preprocessing.py::test_denoise_preserves_dimensions PASSED    [ 82%]
tests/test_preprocessing.py::test_full_preprocessing_pipeline PASSED     [ 88%]
tests/test_spatial_analyzer.py::test_spatial_analyzer_pass_verdict PASSED [ 94%]
tests/test_spatial_analyzer.py::test_spatial_analyzer_fail_verdict_on_crack PASSED [100%]

============================= 17 passed in 4.73s ==============================
```

---

## 📊 Sample Visual Artifacts

For each inspection, VisionInspect automatically generates:
1. **Diagnostic Composite Card (`*_diagnostic_card.png`)**: A 4-panel analysis card showing:
   - *Panel 1*: Raw surface input.
   - *Panel 2*: Radiometrically normalized and CLAHE-equalized image.
   - *Panel 3*: Jet colormap thermal anomaly energy heatmap.
   - *Panel 4*: Quality decision overlay with color-coded bounding boxes and contour masks.
2. **Thermal Heatmap (`*_heatmap.png`)**: Continuous thermal visualization of defect gradient energy.
3. **Inspection Certificate (`*_certificate.md`)**: Human-readable Markdown certificate listing full metrology telemetry.
4. **Audit Record (`*_inspection.json`)**: Machine-readable JSON telemetry for database integration.

---

## 📜 Project Report

A complete 15-section academic project report adhering strictly to the VITyarthi course evaluation rubric is available in:
- Markdown Source: [`report.md`](report.md)
- Printable HTML generator: `python scripts/export_report_html.py`

---

## ⚖️ License
This project is licensed under the MIT License - see the LICENSE file for details.
