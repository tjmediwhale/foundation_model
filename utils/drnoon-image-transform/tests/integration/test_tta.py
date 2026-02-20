from typing import List, Optional

import numpy as np
import pytest

import drnoon_image_transform as dit
from tests.helpers import load_rgb_image, save_rgb_image

TARGET_SHAPE = (100, 100)


def _run_tta(
    image: np.ndarray,
    param_dict: dict,
    target_shape: Optional[tuple] = None,
) -> None:
    """Run TTA on an image."""

    param = dit.TTAParam(**param_dict)
    tta = dit.TTA(param=param, target_shape=target_shape)
    transformed_images = tta(image)
    return transformed_images


@pytest.mark.parametrize(
    "random_image",
    [(10, 10), (10, 20), (20, 10)],
    indirect=True,
)
def test_tta_identity_random(random_image: np.ndarray) -> None:
    """Test that the identity TTA does not change the random image."""

    transformed_images = _run_tta(random_image, {})

    assert len(transformed_images) == 1
    assert np.allclose(transformed_images[0], random_image)


@pytest.mark.parametrize(
    "random_image",
    [(10, 10), (10, 20), (20, 10)],
    indirect=True,
)
@pytest.mark.parametrize(
    "param_dict, expected_n_transforms",
    [
        ({"scale": [0.7, 1, 1.3]}, 3),
        ({"aspect": [0.7, 1, 1.3]}, 3),
        ({"rotate": [-45, 0, 45]}, 3),
        ({"translate_x": [-0.3, 0, 0.3]}, 3),
        ({"translate_y": [-0.3, 0, 0.3]}, 3),
        ({"shear_x": [-15, 0, 15]}, 3),
        ({"shear_y": [-15, 0, 15]}, 3),
        ({"hflip": [True, False]}, 2),
        ({"vflip": [True, False]}, 2),
        ({"brightness": [-0.3, 0, 0.3]}, 3),
        ({"contrast": [-0.3, 0, 0.3]}, 3),
        (
            {
                "scale": [0.7, 1, 1.3],
                "aspect": [0.7, 1, 1.3],
                "rotate": [-45, 0, 45],
            },
            3 * 3 * 3,
        ),
        (
            {
                "translate_x": [-0.3, 0, 0.3],
                "translate_y": [-0.3, 0, 0.3],
                "shear_x": [-15, 0, 15],
                "shear_y": [-15, 0, 15],
            },
            3 * 3 * 3 * 3,
        ),
        (
            {
                "hflip": [True, False],
                "vflip": [True, False],
                "brightness": [-0.3, 0, 0.3],
                "contrast": [-0.3, 0, 0.3],
            },
            2 * 2 * 3 * 3,
        ),
    ],
)
def test_tta_n_transforms(
    random_image: np.ndarray,
    param_dict: dit.TTAParam,
    expected_n_transforms: int,
) -> None:
    """Test that TTA generates the expected number of images."""

    transformed_images = _run_tta(
        image=random_image,
        param_dict=param_dict,
        target_shape=TARGET_SHAPE,
    )

    assert len(transformed_images) == expected_n_transforms


@pytest.mark.parametrize("image_category", ["fundus", "wide"])
@pytest.mark.parametrize(
    "expected_dir, param_dict",
    [
        (
            "scale=[0.7,1.3]_rotate=[-45,0,45]_flip=[False,True]",
            {
                "scale": [0.7, 1.3],
                "rotate": [-45, 0, 45],
                "hflip": [False, True],
            },
        ),
    ],
)
def test_tta(
    pytestconfig,  # Injected by pytest to access configuration and command-line options
    image_category: str,
    expected_dir: List[str],
    param_dict: dict,
) -> None:
    """Test that TTA generates the expected images."""

    # Adjust the paths based on the image category
    original_image_path = f"{image_category}/original.png"
    expected_image_dir = f"{image_category}/tta/{expected_dir}"

    # Load the original image
    image = load_rgb_image(original_image_path)

    # Run the TTA
    image = load_rgb_image(original_image_path)
    transformed_images = _run_tta(
        image=image,
        param_dict=param_dict,
        target_shape=TARGET_SHAPE,
    )

    for i, transformed in enumerate(transformed_images):
        expected_image_path = f"{expected_image_dir}/{i:02d}.png"
        if pytestconfig.getoption("--save-images"):
            # Save the transformed image
            save_rgb_image(transformed, expected_image_path)
        else:
            # Load the expected image and compare
            expected = load_rgb_image(expected_image_path)
            assert np.allclose(transformed, expected)
