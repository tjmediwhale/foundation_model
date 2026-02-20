import sys
from pathlib import Path

import cv2
import numpy as np

FIXTURE_HOME = Path("tests/fixtures")


def get_python_version() -> str:
    """Get the Python version (major.minor)."""

    return f"{sys.version_info.major}.{sys.version_info.minor}"


def load_rgb_image(image_path: str) -> np.ndarray:
    """Load a RGB image from a fixture file.

    Args:
        image_path: The path to the image file relative to the fixture home.

    Returns:
        The RGB image array.
    """

    image_path = FIXTURE_HOME / image_path
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def save_rgb_image(image: np.ndarray, image_path: str) -> None:
    """Save a RGB image to a fixture file.

    Args:
        image: The RGB image array.
        image_path: The path to the image file relative to the fixture home.
    """

    image_path = FIXTURE_HOME / image_path
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_path), image)
