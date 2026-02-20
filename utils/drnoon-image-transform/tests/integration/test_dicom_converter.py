import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pydicom
import pytest

from drnoon_image_transform import DicomConverter

FIXTURE_DIR = Path("tests/fixtures/convert_dicom")


def get_dicom_files(directory: str) -> List[str]:
    """Get a list of DICOM files in a fixture/convert_dicom directory."""
    return sorted([f for f in os.listdir(directory) if f.endswith(".dcm")])


def get_corresponding_png_file(dicom_file: str) -> str:
    """Get the corresponding PNG file for a DICOM file."""
    return dicom_file.replace(".dcm", ".png")


@pytest.fixture(params=get_dicom_files(FIXTURE_DIR))
def dicom_and_png(request) -> Tuple[str, str]:
    """Fixture to provide a DICOM file and its corresponding PNG file."""
    dicom_file = request.param
    dicom_path = os.path.join(FIXTURE_DIR, dicom_file)
    png_file = get_corresponding_png_file(dicom_file)
    png_path = os.path.join(FIXTURE_DIR, png_file)
    return dicom_path, png_path


@pytest.fixture
def converter() -> DicomConverter:
    """Fixture to provide a DicomConverter instance."""
    return DicomConverter()


def test_convert_dicom_to_image_array(
    pytestconfig, converter: DicomConverter, dicom_and_png: Tuple[str, str]
) -> None:
    """Test the conversion of a DICOM file to an image array.

    Compare to np.ndarray from dicom conversion and existing png file.
    """
    dicom_path, png_path = dicom_and_png

    dicom_dataset = pydicom.dcmread(dicom_path)

    image_array = converter(dicom_dataset)

    expected_array = cv2.imread(png_path)
    expected_array = cv2.cvtColor(expected_array, cv2.COLOR_BGR2RGB)

    assert isinstance(image_array, np.ndarray), "The result should be a numpy array"
    assert image_array.ndim == 3, "The image should be a 3-dimensional array"
    assert image_array.shape[2] == 3, "The image should have 3 channels for RGB"

    assert (
        image_array.shape[0] > 0 and image_array.shape[1] > 0
    ), "Image dimensions should be positive"
    assert image_array.dtype == np.uint8, "Image data type should be np.uint8"

    assert (
        np.min(image_array) >= 0 and np.max(image_array) <= 255
    ), "Pixel values should be in the range 0-255"

    assert (
        image_array.shape == expected_array.shape
    ), f"Shapes do not match for {dicom_path} and {png_path}"
    assert np.allclose(
        image_array, expected_array
    ), f"Arrays do not match for {dicom_path} and {png_path}"
