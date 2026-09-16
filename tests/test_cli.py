"""
Integration tests for CLI commands (inspect, batch, benchmark).
"""

import subprocess
import sys
from pathlib import Path
import pytest


PYTHON_EXE = sys.executable
PROJECT_ROOT = Path(__file__).parent.parent


def test_cli_help():
    result = subprocess.run(
        [PYTHON_EXE, "-m", "vision_inspect.cli", "--help"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "VisionInspect" in result.stdout
    assert "inspect" in result.stdout
    assert "batch" in result.stdout


def test_cli_inspect_clean():
    sample_img = PROJECT_ROOT / "data" / "samples" / "sample_01_clean.png"
    if not sample_img.exists():
        pytest.skip("Sample image not found")

    result = subprocess.run(
        [PYTHON_EXE, "-m", "vision_inspect.cli", "inspect", "--input", str(sample_img), "--output-dir", "output_test"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "INSPECTION RESULT: [PASS]" in result.stdout


def test_cli_inspect_crack():
    crack_img = PROJECT_ROOT / "data" / "samples" / "sample_03_crack.png"
    if not crack_img.exists():
        pytest.skip("Sample image not found")

    result = subprocess.run(
        [PYTHON_EXE, "-m", "vision_inspect.cli", "inspect", "--input", str(crack_img), "--output-dir", "output_test"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True
    )
    assert result.returncode == 1  # Exit code 1 indicates rejected product
    assert "INSPECTION RESULT: [FAIL - REJECTED]" in result.stdout
    assert "CRITICAL" in result.stdout
