"""
Configuration dataclasses and settings for the VisionInspect pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Optional
import json


@dataclass
class PreprocessingConfig:
    """Parameters for radiometric correction, noise removal, and contrast enhancement."""
    clahe_clip_limit: float = 2.5
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    illumination_kernel_size: int = 45  # Large morphological opening kernel for background estimation
    target_size: Optional[Tuple[int, int]] = (640, 640)  # Standardized analysis resolution


@dataclass
class ClassicalDetectionConfig:
    """Parameters for edge detection, adaptive thresholding, and morphological operators."""
    adaptive_block_size: int = 15
    adaptive_c: int = 4
    morph_kernel_size: int = 3
    morph_iterations: int = 2
    gradient_ksize: int = 3
    gradient_threshold: int = 35
    min_defect_area: int = 25       # Minimum pixel area to filter micro-noise
    max_defect_area: int = 50000    # Maximum pixel area to filter global lighting artifacts


@dataclass
class TextureFeatureConfig:
    """Parameters for Gray-Level Co-occurrence Matrix (GLCM) and local statistical textures."""
    glcm_distances: List[int] = field(default_factory=lambda: [1, 3, 5])
    glcm_angles: List[float] = field(default_factory=lambda: [0, 0.785398, 1.570796, 2.356194]) # 0, 45, 90, 135 deg
    patch_size: int = 32
    stride: int = 16
    z_score_threshold: float = 2.8


@dataclass
class SeverityConfig:
    """Thresholds for industrial defect severity classification."""
    # Pixel area thresholds
    minor_max_area: int = 80
    major_max_area: int = 350
    # Aspect ratio threshold for classifying linear defects (cracks, long scratches)
    crack_aspect_ratio_threshold: float = 3.5
    critical_defect_types: List[str] = field(default_factory=lambda: ["crack", "void", "deep_scratch"])


@dataclass
class InspectionConfig:
    """Master configuration encompassing all pipeline modules."""
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    classical: ClassicalDetectionConfig = field(default_factory=ClassicalDetectionConfig)
    texture: TextureFeatureConfig = field(default_factory=TextureFeatureConfig)
    severity: SeverityConfig = field(default_factory=SeverityConfig)
    
    # Quality gate: Maximum permissible defect count or area for a PASS verdict
    max_tolerated_defects: int = 0
    max_tolerated_defect_area: int = 0
    
    # Visualization & reporting
    save_visualizations: bool = True
    save_heatmaps: bool = True
    generate_json: bool = True
    generate_csv: bool = True

    def to_json(self, filepath: Path) -> None:
        """Serializes configuration to JSON file."""
        data = {
            "preprocessing": self.preprocessing.__dict__,
            "classical": self.classical.__dict__,
            "texture": self.texture.__dict__,
            "severity": self.severity.__dict__,
            "max_tolerated_defects": self.max_tolerated_defects,
            "max_tolerated_defect_area": self.max_tolerated_defect_area,
            "save_visualizations": self.save_visualizations,
            "save_heatmaps": self.save_heatmaps,
            "generate_json": self.generate_json,
            "generate_csv": self.generate_csv
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
