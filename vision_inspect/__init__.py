"""
VisionInspect: Automated Industrial Surface Defect Detection & Quality Inspection System.
"""

__version__ = "1.0.0"
__author__ = "VisionInspect Team"

from vision_inspect.config import InspectionConfig
from vision_inspect.preprocessing import Preprocessor
from vision_inspect.classical_engine import MorphologicalEngine
from vision_inspect.feature_extractor import TextureFeatureExtractor
from vision_inspect.anomaly_detector import AnomalyDetector
from vision_inspect.spatial_analyzer import SpatialAnalyzer, DefectMetric
from vision_inspect.visualizer import Visualizer
from vision_inspect.reporter import InspectionReporter

__all__ = [
    "InspectionConfig",
    "Preprocessor",
    "MorphologicalEngine",
    "TextureFeatureExtractor",
    "AnomalyDetector",
    "SpatialAnalyzer",
    "DefectMetric",
    "Visualizer",
    "InspectionReporter",
]
