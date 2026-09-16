"""
Procedural Industrial Surface and Defect Generator.
Generates realistic textured metal/ceramic surfaces with ground-truth defect masks
for rigorous benchmarking and automated testing.
"""

from pathlib import Path
from typing import Tuple
import random
import cv2
import numpy as np


class SyntheticSurfaceGenerator:
    """
    Synthesizes metallic and composite surface textures with controlled,
    parameterized physical defects (cracks, scratches, pitting, blemishes).
    """

    def __init__(self, size: Tuple[int, int] = (640, 640), seed: int = 42):
        self.width, self.height = size
        np.random.seed(seed)
        random.seed(seed)

    def generate_base_surface(self, base_gray: int = 180, roughness: float = 8.0) -> np.ndarray:
        """
        Creates a brushed metal/cast substrate texture with subtle Perlin-like noise
        and non-uniform lighting gradient.
        """
        # Base uniform plane
        surface = np.full((self.height, self.width), base_gray, dtype=np.float32)

        # Micro-texture noise (brushed metal effect: horizontal streaks)
        noise = np.random.normal(0, roughness, (self.height, self.width)).astype(np.float32)
        # Blur horizontally to simulate brushed grain
        noise_brushed = cv2.GaussianBlur(noise, (15, 3), 0)
        surface += noise_brushed

        # Add gentle non-uniform vignetting illumination gradient
        y, x = np.ogrid[:self.height, :self.width]
        cx, cy = self.width / 2.0, self.height / 2.0
        dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        illumination_falloff = 1.0 - 0.25 * (dist_from_center / max_dist) ** 2
        surface *= illumination_falloff

        return np.clip(surface, 0, 255).astype(np.uint8)

    def inject_crack(self, surface: np.ndarray, mask: np.ndarray) -> None:
        """
        Synthesizes a realistic branching, jagged hairline crack using a constrained random walk.
        """
        start_x = random.randint(self.width // 4, 3 * self.width // 4)
        start_y = random.randint(self.height // 4, 3 * self.height // 4)
        steps = random.randint(120, 220)

        curr_x, curr_y = float(start_x), float(start_y)
        angle = random.uniform(0, 2 * np.pi)

        pts = [(int(curr_x), int(curr_y))]

        for _ in range(steps):
            angle += random.uniform(-0.35, 0.35)
            step_len = random.uniform(2.0, 4.0)
            curr_x += step_len * np.cos(angle)
            curr_y += step_len * np.sin(angle)

            if 10 <= curr_x < self.width - 10 and 10 <= curr_y < self.height - 10:
                pts.append((int(curr_x), int(curr_y)))
            else:
                break

        for i in range(len(pts) - 1):
            thickness = random.randint(2, 3)
            # Crack is darker than background
            cv2.line(surface, pts[i], pts[i + 1], color=35, thickness=thickness)
            cv2.line(mask, pts[i], pts[i + 1], color=255, thickness=thickness + 1)

    def inject_scratch(self, surface: np.ndarray, mask: np.ndarray) -> None:
        """
        Synthesizes a straight or gently curved machining scratch.
        """
        p1 = (random.randint(50, self.width - 50), random.randint(50, self.height - 50))
        angle = random.uniform(0, np.pi)
        length = random.randint(80, 200)
        p2 = (
            int(p1[0] + length * np.cos(angle)),
            int(p1[1] + length * np.sin(angle))
        )

        thickness = random.randint(2, 4)
        cv2.line(surface, p1, p2, color=50, thickness=thickness)
        cv2.line(mask, p1, p2, color=255, thickness=thickness)

    def inject_pitting(self, surface: np.ndarray, mask: np.ndarray) -> None:
        """
        Synthesizes corrosion pits and cavitation voids (clusters of dark circular indentations).
        """
        center_x = random.randint(100, self.width - 100)
        center_y = random.randint(100, self.height - 100)
        cluster_count = random.randint(4, 9)

        for _ in range(cluster_count):
            ox = center_x + random.randint(-35, 35)
            oy = center_y + random.randint(-35, 35)
            radius = random.randint(4, 10)
            if 0 <= ox < self.width and 0 <= oy < self.height:
                cv2.circle(surface, (ox, oy), radius, color=30, thickness=-1)
                cv2.circle(mask, (ox, oy), radius, color=255, thickness=-1)

    def inject_blemish(self, surface: np.ndarray, mask: np.ndarray) -> None:
        """
        Synthesizes an irregular discoloration or surface oxidation patch.
        """
        cx = random.randint(120, self.width - 120)
        cy = random.randint(120, self.height - 120)
        axes = (random.randint(25, 45), random.randint(15, 30))
        angle = random.randint(0, 180)

        patch_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.ellipse(patch_mask, (cx, cy), axes, angle, 0, 360, 255, -1)
        patch_mask = cv2.GaussianBlur(patch_mask, (21, 21), 0)

        # Attenuate surface intensity in blemish region
        blemish_factor = (patch_mask.astype(np.float32) / 255.0) * 0.45
        blended = surface.astype(np.float32) * (1.0 - blemish_factor)
        surface[:] = np.clip(blended, 0, 255).astype(np.uint8)
        cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, -1)

    def generate_dataset(self, output_dir: Path, gt_dir: Path) -> None:
        """
        Generates a standard benchmark suite of clean and defective test specimens.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

        samples_meta = [
            ("sample_01_clean", []),
            ("sample_02_clean", []),
            ("sample_03_crack", ["crack"]),
            ("sample_04_crack_multi", ["crack", "crack"]),
            ("sample_05_scratch", ["scratch"]),
            ("sample_06_pitting", ["pitting"]),
            ("sample_07_complex_defects", ["crack", "pitting"]),
            ("sample_08_blemish", ["blemish"]),
        ]

        for name, defects in samples_meta:
            surface = self.generate_base_surface()
            mask = np.zeros((self.height, self.width), dtype=np.uint8)

            for d_type in defects:
                if d_type == "crack":
                    self.inject_crack(surface, mask)
                elif d_type == "scratch":
                    self.inject_scratch(surface, mask)
                elif d_type == "pitting":
                    self.inject_pitting(surface, mask)
                elif d_type == "blemish":
                    self.inject_blemish(surface, mask)

            # Convert surface to BGR for realistic 3-channel input
            surface_bgr = cv2.cvtColor(surface, cv2.COLOR_GRAY2BGR)

            img_file = output_dir / f"{name}.png"
            mask_file = gt_dir / f"{name}_mask.png"

            cv2.imwrite(str(img_file), surface_bgr)
            cv2.imwrite(str(mask_file), mask)
            print(f"[+] Generated: {img_file.name} (Defects: {defects or 'None (Clean)'})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic surface test dataset")
    parser.add_argument("--samples-dir", default="data/samples", help="Output directory for sample images")
    parser.add_argument("--gt-dir", default="data/ground_truth", help="Output directory for ground truth masks")
    args = parser.parse_args()

    gen = SyntheticSurfaceGenerator()
    gen.generate_dataset(Path(args.samples_dir), Path(args.gt_dir))
    print("[*] Dataset generation complete.")
