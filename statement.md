# Project Statement: VisionInspect

## 1. Problem Statement
In high-precision manufacturing and industrial production lines (such as aerospace alloy machining, semiconductor wafer fabrication, automotive sheet-metal stamping, and ceramic component casting), surface flaws including micro-cracks, mechanical abrasions/scratches, cavitation pitting, and material inclusions directly compromise structural integrity and component longevity.

Traditional quality control relies heavily on human visual inspection, which suffers from:
- High operator fatigue and subjective variance, leading to inconsistent defect classification.
- Low throughput that limits 100% inline quality verification on high-speed conveyor systems.
- High cost and inability to log quantitative spatial telemetry (e.g., exact micron/pixel defect area, aspect ratio, circularity, and ASTM/ISO tolerance thresholds) into digitized audit trails.

Existing commercial machine vision solutions are often proprietary, cost-prohibitive, tightly bound to dedicated vendor cameras, and lack headless, cross-platform Command-Line Interface (CLI) workflows suitable for automated continuous integration (CI) and edge-embedded industrial computers.

**VisionInspect** resolves this problem by delivering an open, modular, highly deterministic Computer Vision pipeline capable of real-time surface defect detection, sub-pixel contour localization, statistical texture analysis, geometric defect typing, and automated industrial severity grading through a lightweight command-line interface.

---

## 2. Scope of the Project
The scope of **VisionInspect** encompasses:

- **Radiometric Normalization & Enhancement**: Dynamic morphological illumination correction to eliminate non-uniform lighting and shadows, bilateral edge-preserving smoothing, and Contrast Limited Adaptive Histogram Equalization (CLAHE).
- **Multi-Cue Classical Defect Detection**: Fusion of Sobel gradient magnitudes, adaptive Gaussian thresholding, and morphological operators to detect hairline cracks, scratches, voids, and blemishes.
- **Statistical Texture & Morphometric Analysis**: Second-order Gray-Level Co-occurrence Matrix (GLCM) statistical features (contrast, dissimilarity, homogeneity, energy, correlation) and shape factors (aspect ratio, circularity, solidity, extent) to categorize defect types.
- **Spatial Metrology & Quality Decision Gates**: Pixel area measurement, equivalent circular diameter calculation, defect coverage percentage, and rule-based severity grading (`CRITICAL`, `MAJOR`, `MINOR`) to deliver automated `PASS` or `FAIL` quality verdicts.
- **Command-Line Interface (CLI) & Automated Auditing**: Standalone execution without GUI dependencies, supporting single-image inspection, directory-wide batch processing, benchmark evaluation against ground-truth masks, and export to JSON, CSV audit logs, and Markdown inspection certificates.
- **Procedural Specimen Synthesis**: Built-in benchmark generator for synthesizing realistic metallic surfaces with controlled hairline cracks, scratches, corrosion clusters, and ground-truth segmentation masks.

*Out of Scope*: Physical conveyor robotic arm integration, programmable logic controller (PLC) hardware bus protocols (such as Modbus/Profibus), and proprietary camera sensor SDKs.

---

## 3. Target Users
1. **Quality Assurance (QA) & Quality Control (QC) Engineers**: Engineers monitoring industrial production lines who require instant, deterministic defect reports, pass/fail decisions, and audit records.
2. **Manufacturing & Industrial Automation Teams**: Teams deploying edge computing systems (e.g., Raspberry Pi, NVIDIA Jetson, industrial x86 PCs) mounted above inspection conveyor belts.
3. **Computer Vision Researchers & Students**: Academics and developers benchmarking segmentation algorithms, morphological operators, and texture descriptors on standardized defect datasets.
4. **Maintenance, Repair, and Overhaul (MRO) Technicians**: Inspectors conducting non-destructive surface evaluations (NDE/NDT) on turbines, pipelines, and structural assemblies.

---

## 4. High-Level Features
- **Headless CLI Execution**: Operates completely in terminal environments with rich status telemetry and deterministic exit codes (`0` for PASS, `1` for FAIL/Rejection, `2` for runtime error).
- **Adaptive Illumination Compensation**: Resilient to non-uniform ambient light and shadows through morphological background estimation.
- **Hybrid Detection Architecture**: Integrates differential edge operators, adaptive neighborhood binarization, and GLCM texture statistics.
- **Geometric Defect Classification**: Automatically categorizes detected anomalies into structural `Crack`, abrasive `Scratch`, localized `Pitting`, or discolored `Blemish`.
- **Industrial Severity Grading**: Quantifies risk into `CRITICAL` (structural failure risk), `MAJOR` (functional deviation), and `MINOR` (cosmetic flaw).
- **Multi-Format Audit Export**: Generates JSON telemetry records, CSV line summaries, Markdown QA certificates, and color-coded overlay diagnostic cards.
- **Quantitative Benchmark Runner**: Evaluates precision, recall, F1-score, and Intersection over Union (IoU) against ground-truth masks with sub-50ms CPU execution speeds.
