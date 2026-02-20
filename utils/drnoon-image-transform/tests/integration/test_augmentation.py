import random
from typing import Optional

import numpy as np
import pytest

import drnoon_image_transform as dit
from tests.helpers import get_python_version, load_rgb_image, save_rgb_image

SEED = 1106
TARGET_SHAPE = (100, 100)
N_AUGS_SINGLE = 3
N_AUGS_COMBINED = 10


def _generate_augmentation(
    image: np.ndarray,
    param_dict: dict,
    target_shape: Optional[tuple] = None,
    normalize: Optional[dict] = None,
    n_augs: int = 1,
) -> None:
    """Generate a random augmentation using the given parameters."""

    np.random.seed(SEED)
    random.seed(SEED)

    param = dit.RandomAugParam(**param_dict)
    transform = dit.RandomAugmentation(param=param, target_shape=target_shape, normalize=normalize)
    for _ in range(n_augs):
        yield transform(image)


def _test_augmentation(
    pytestconfig,
    image_path: str,
    expected_dir: str,
    param_dict: dict,
    n_augs: int,
) -> None:
    """Test a random augmentation using the given parameters."""

    # Load the original image
    image = load_rgb_image(image_path)

    # Run the augmentation
    for i, transformed in enumerate(
        _generate_augmentation(
            image=image,
            param_dict=param_dict,
            target_shape=TARGET_SHAPE,
            n_augs=n_augs,
        )
    ):
        expected_image_path = f"{expected_dir}/{i}.png"

        # Check the shape of the transformed image
        assert transformed.shape[:2] == TARGET_SHAPE

        # The original image shuld not be modified
        assert np.allclose(image, load_rgb_image(image_path))

        # Save images or compare with saved images depending on the command-line option
        if pytestconfig.getoption("--save-images"):
            # Save the transformed image
            save_rgb_image(transformed, expected_image_path)

        else:
            # Load the expected image and compare
            expected = load_rgb_image(expected_image_path)

            if get_python_version() != "3.7":
                # HACK: Looks like various albumentation transforms are not deterministic across Python versions.
                # This is probably due to the underlying OpenCV implementation.
                # Just replace with very loose checks (mean and std with large tolerances).
                np.isclose(transformed.mean(), expected.mean(), atol=10)
                np.isclose(transformed.std(), expected.std(), atol=10)
                return

            assert np.allclose(transformed, expected, atol=1)


@pytest.mark.parametrize("random_image", [(10, 10), (10, 20), (20, 10)], indirect=True)
def test_identity_random(random_image: np.ndarray) -> None:
    """Test that the identity transform does not change the random image."""

    for transformed in _generate_augmentation(random_image, {}):
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

    for transformed in _generate_augmentation(random_image, param_dict={}, normalize=normalize):
        mean = np.array(normalize["mean"]) * 255.0
        std = np.array(normalize["std"]) * 255.0
        expected = (random_image - mean) / std
        assert np.allclose(transformed, expected)


@pytest.mark.parametrize("image_category", ["fundus", "wide"])
@pytest.mark.parametrize(
    "aug_category, aug_type",
    [
        ("geometric", "scale"),
        ("geometric", "aspect"),
        ("geometric", "rotate"),
        ("geometric", "translate_x"),
        ("geometric", "translate_y"),
        ("geometric", "shear_x"),
        ("geometric", "shear_y"),
        ("geometric", "hflip"),
        ("geometric", "vflip"),
        ("photometric", "color_jitter"),
        ("photometric", "random_brightness_contrast"),
        ("photometric", "random_gamma"),
        ("photometric", "blur"),
        ("photometric", "gaussian_blur"),
        ("photometric", "median_blur"),
        ("photometric", "motion_blur"),
        ("photometric", "zoom_blur"),
        ("photometric", "defocus"),
        ("photometric", "downscale"),
        ("photometric", "image_compression"),
        ("photometric", "posterize"),
        ("photometric", "solarize"),
        ("photometric", "sharpen"),
        ("photometric", "equalize"),
        ("photometric", "clahe"),
        ("photometric", "gauss_noise"),
        ("photometric", "iso_noise"),
        ("photometric", "multiplicative_noise"),
        ("photometric", "gaussian_blackout"),
        ("photometric", "fundus_contrast_enhancement"),
        ("preaug", "precrop"),
        ("preaug", "circle_mask"),
        ("postaug", "coarse_dropout"),
    ],
)
def test_single_type_augmentation(
    pytestconfig,
    image_category: str,
    aug_category: str,
    aug_type: str,
) -> None:
    """Test a single type augmentation."""

    # Adjust the paths based on the image category
    original_image_path = f"{image_category}/original.png"
    expected_image_dir = f"{image_category}/augmentations/{aug_type}"

    # Set the parameters
    if image_category == "fundus":
        predefined_param = dit.FUNDUS_RANDOM_AUG_PARAM
    elif image_category == "wide":
        predefined_param = dit.WIDE_RANDOM_AUG_PARAM
    aug_param = getattr(predefined_param, aug_category)
    aug_param = getattr(aug_param, aug_type)
    if aug_category in ["photometric", "postaug"]:
        aug_param = {**aug_param, "p": 1.0}
    param_dict = {aug_category: {aug_type: aug_param}}

    # Test the augmentation
    _test_augmentation(
        pytestconfig=pytestconfig,
        image_path=original_image_path,
        expected_dir=expected_image_dir,
        param_dict=param_dict,
        n_augs=N_AUGS_SINGLE,
    )


@pytest.mark.parametrize("image_category", ["fundus", "wide"])
def test_combined_augmentation(pytestconfig, image_category: str) -> None:
    """Test a combined random augmentation."""

    # Adjust the paths based on the image category
    original_image_path = f"{image_category}/original.png"
    expected_image_dir = f"{image_category}/augmentations/combined"

    # Set the parameters
    if image_category == "fundus":
        predefined_param = dit.FUNDUS_RANDOM_AUG_PARAM
    elif image_category == "wide":
        predefined_param = dit.WIDE_RANDOM_AUG_PARAM
    param_dict = predefined_param.dict()

    # Test the augmentation
    _test_augmentation(
        pytestconfig=pytestconfig,
        image_path=original_image_path,
        expected_dir=expected_image_dir,
        param_dict=param_dict,
        n_augs=N_AUGS_COMBINED,
    )
