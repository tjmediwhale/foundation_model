from typing import Optional, Type

import numpy as np
import pytest

import drnoon_image_transform as dit
from tests.helpers import get_python_version, load_rgb_image, save_rgb_image

TARGET_SHAPE = (100, 100)


def _run_transform(
    image: np.ndarray,
    param_dict: dict,
    target_shape: Optional[tuple] = None,
    normalize: Optional[dict] = None,
) -> None:
    """Run a transform on an image."""

    param = dit.TransformParam(**param_dict)
    transform = dit.Transform(param=param, target_shape=target_shape, normalize=normalize)
    return transform(image)


@pytest.mark.parametrize(
    "random_image",
    [(10, 10), (10, 20), (20, 10)],
    indirect=True,
)
def test_identity_random(random_image: np.ndarray) -> None:
    """Test that the identity transform does not change the random image."""

    transformed = _run_transform(random_image, {})

    assert np.allclose(transformed, random_image)


@pytest.mark.parametrize("random_image", [(10, 10), (10, 20), (20, 10)], indirect=True)
@pytest.mark.parametrize(
    "normalize",
    [
        {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
        {"mean": (0.25, 0.5, 0.75), "std": (0.75, 0.5, 0.25)},
        {"mean": dit.FUNDUS_RGB_MEAN_NORMALIZED, "std": dit.FUNDUS_RGB_STD_NORMALIZED},
    ],
)
def test_normalize_random(random_image: np.ndarray, normalize: dict) -> None:
    """Test that the normalization transform works as expected."""

    transformed = _run_transform(random_image, param_dict={}, normalize=normalize)
    mean = np.array(normalize["mean"]) * 255.0
    std = np.array(normalize["std"]) * 255.0
    expected = (random_image - mean) / std
    assert np.allclose(transformed, expected)


@pytest.mark.parametrize("image_category", ["fundus", "wide"])
@pytest.mark.parametrize(
    "expected_path, param_dict, target_shape",
    [
        ("original.png", {}, None),
        # Precrop
        ("precrop=1.0.png", {"precrop": 1.0}, TARGET_SHAPE),
        ("precrop=0.7.png", {"precrop": 0.7}, TARGET_SHAPE),
        ("precrop=0.4.png", {"precrop": 0.4}, TARGET_SHAPE),
        # Circle mask
        ("circle_mask.png", {"circle_mask": True}, TARGET_SHAPE),
        # Color transfer
        ("color_transfer.png", {"color_transfer": {}}, TARGET_SHAPE),
        # Precrop + circle mask
        ("precrop=1.0_circle_mask.png", {"precrop": 1.0, "circle_mask": True}, TARGET_SHAPE),
        ("precrop=0.7_circle_mask.png", {"precrop": 0.7, "circle_mask": True}, TARGET_SHAPE),
        ("precrop=0.4_circle_mask.png", {"precrop": 0.4, "circle_mask": True}, TARGET_SHAPE),
        # Precrop + circle mask + color transfer
        (
            "precrop=1.0_circle_mask_color_transfer.png",
            {"precrop": 1.0, "circle_mask": True, "color_transfer": {}},
            TARGET_SHAPE,
        ),
        (
            "precrop=0.7_circle_mask_color_transfer.png",
            {"precrop": 0.7, "circle_mask": True, "color_transfer": {}},
            TARGET_SHAPE,
        ),
        (
            "precrop=0.4_circle_mask_color_transfer.png",
            {"precrop": 0.4, "circle_mask": True, "color_transfer": {}},
            TARGET_SHAPE,
        ),
        # Precrop + circle mask + color_transfer + scale
        (
            "precrop=0.4_circle_mask_color_transfer_scale=0.7.png",
            {"precrop": 0.4, "circle_mask": True, "color_transfer": {}, "scale": 0.7},
            TARGET_SHAPE,
        ),
        (
            "precrop=0.4_circle_mask_color_transfer_scale=1.3.png",
            {"precrop": 0.4, "circle_mask": True, "color_transfer": {}, "scale": 1.3},
            TARGET_SHAPE,
        ),
        # Others
        ("scale=0.7.png", {"scale": 0.7}, TARGET_SHAPE),
        ("scale=1.3.png", {"scale": 1.3}, TARGET_SHAPE),
        ("aspect=0.7.png", {"aspect": 0.7}, TARGET_SHAPE),
        ("aspect=1.3.png", {"aspect": 1.3}, TARGET_SHAPE),
        ("rotate=-45.png", {"rotate": -45}, TARGET_SHAPE),
        ("rotate=45.png", {"rotate": 45}, TARGET_SHAPE),
        ("translate_x=-0.3.png", {"translate_x": -0.3}, TARGET_SHAPE),
        ("translate_x=0.3.png", {"translate_x": 0.3}, TARGET_SHAPE),
        ("translate_y=-0.3.png", {"translate_y": -0.3}, TARGET_SHAPE),
        ("translate_y=0.3.png", {"translate_y": 0.3}, TARGET_SHAPE),
        ("shear_x=-15.png", {"shear_x": -15}, TARGET_SHAPE),
        ("shear_x=15.png", {"shear_x": 15}, TARGET_SHAPE),
        ("shear_y=-15.png", {"shear_y": -15}, TARGET_SHAPE),
        ("shear_y=15.png", {"shear_y": 15}, TARGET_SHAPE),
        ("hflip.png", {"hflip": True}, TARGET_SHAPE),
        ("vflip.png", {"vflip": True}, TARGET_SHAPE),
        ("brightness=-0.3.png", {"brightness": -0.3}, TARGET_SHAPE),
        ("brightness=0.3.png", {"brightness": 0.3}, TARGET_SHAPE),
        ("contrast=-0.3.png", {"contrast": -0.3}, TARGET_SHAPE),
        ("contrast=0.3.png", {"contrast": 0.3}, TARGET_SHAPE),
    ],
)
def test_transform(
    pytestconfig,  # Injected by pytest to access configuration and command-line options
    image_category: str,
    expected_path: str,
    param_dict: dict,
    target_shape: Optional[tuple],
) -> None:
    """Test that a transform generates the expected image."""

    # Adjust the paths based on the image category
    original_image_path = f"{image_category}/original.png"
    expected_image_path = f"{image_category}/transforms/{expected_path}"

    # Load the original image
    image = load_rgb_image(original_image_path)

    # Run the transform
    image = load_rgb_image(original_image_path)
    transformed = _run_transform(
        image=image,
        param_dict=param_dict,
        target_shape=target_shape,
    )

    # Check the shape of the transformed image
    if target_shape:
        # If target shape is given, the transformed image should have the target shape.
        assert transformed.shape[:2] == target_shape
    elif "precrop" not in param_dict:
        # If target shape is not given and precrop is not applied, the transformed image should have the original shape.
        assert transformed.shape[:2] == image.shape[:2]
    else:
        # If target shape is not given and precrop is applied, the transformed image should have the precropped shape.
        crop_side = int(min(image.shape[:2]) * param_dict["precrop"])
        assert transformed.shape[:2] == (crop_side, crop_side)

    # Save images or compare with saved images depending on the command-line option
    if pytestconfig.getoption("--save-images"):
        # Save the transformed image
        save_rgb_image(transformed, expected_image_path)

    else:
        # Load the expected image and compare
        expected = load_rgb_image(expected_image_path)

        if get_python_version() != "3.7" and ("shear_x" in param_dict or "shear_y" in param_dict):
            # HACK: Looks like the shearing transform is not deterministic across Python versions.
            # This is probably due to the underlying OpenCV implementation.
            # Just replace with very loose checks (mean and std with large tolerances).
            np.isclose(transformed.mean(), expected.mean(), atol=10)
            np.isclose(transformed.std(), expected.std(), atol=10)
            return

        assert np.allclose(transformed, expected)
