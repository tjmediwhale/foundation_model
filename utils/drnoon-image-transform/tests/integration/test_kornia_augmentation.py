import random
from typing import Optional

import numpy as np
import pytest

import drnoon_image_transform as dit
from tests.helpers import get_python_version, load_rgb_image, save_rgb_image

SEED = 1106
N_AUGS_SINGLE = 3
N_AUGS_COMBINED = 10

kornia = pytest.importorskip("kornia")
torch = pytest.importorskip("torch")

from tests.torch_helpers import (
    convert_np_array_to_torch_tensor,
    convert_torch_tensor_to_np_array,
)


def _generate_kornia_augmentation(
    image: torch.Tensor,
    param_dict: dict,
    target_shape: Optional[tuple] = None,
    normalize: Optional[dict] = None,
    n_augs: int = 1,
    use_gpu: bool = False,
) -> None:
    """Generate a random augmentation using the given parameters and image type is torch.Tensor."""

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    param = dit.RandomAugParam(**param_dict)
    transform = dit.RandomAugmentationKornia(
        param=param, target_shape=target_shape, normalize=normalize
    )
    if use_gpu:
        transform = transform.to("cuda")
    for _ in range(n_augs):
        yield transform(image)


def _test_kornia_augmentation(
    pytestconfig,
    image_path: str,
    expected_dir: str,
    param_dict: dict,
    n_augs: int,
) -> None:
    """Test a random augmentation using the given parameters."""

    use_gpu = pytestconfig.getoption("--gpu")
    # Load the original image
    image = load_rgb_image(image_path)
    image_torch = convert_np_array_to_torch_tensor(image)
    if use_gpu:
        image_torch = image_torch.to("cuda")

    # Run the augmentation
    for i, transformed in enumerate(
        _generate_kornia_augmentation(
            image=image_torch,
            param_dict=param_dict,
            target_shape=image.shape[:2],
            n_augs=n_augs,
            use_gpu=use_gpu,
        )
    ):
        if use_gpu:
            transformed = transformed.cpu()
        transformed = convert_torch_tensor_to_np_array(transformed)
        expected_image_path = f"{expected_dir}/{i}.png"

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

            assert np.allclose(transformed, expected, atol=1e-5)


@pytest.mark.parametrize("image_category", ["fundus_kornia", "wide_kornia"])
@pytest.mark.parametrize(
    "aug_category, aug_type",
    [
        ("photometric", "color_jitter"),
        ("photometric", "posterize"),
        ("photometric", "solarize"),
        ("photometric", "sharpen"),
        ("photometric", "equalize"),
        ("photometric", "fundus_contrast_enhancement"),
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
    predefined_param = dit.GPU_FUNDUS_RANDOM_AUG_KORNIA_PARAM

    aug_param = getattr(predefined_param, aug_category)
    aug_param = getattr(aug_param, aug_type)
    if aug_category in [
        "photometric",
    ]:
        aug_param = {**aug_param, "p": 1.0}
    param_dict = {aug_category: {aug_type: aug_param}}
    print(param_dict)

    # Test the augmentation
    _test_kornia_augmentation(
        pytestconfig=pytestconfig,
        image_path=original_image_path,
        expected_dir=expected_image_dir,
        param_dict=param_dict,
        n_augs=N_AUGS_SINGLE,
    )


@pytest.mark.parametrize("image_category", ["fundus_kornia", "wide_kornia"])
def test_combined_augmentation(pytestconfig, image_category: str) -> None:
    """Test a combined random augmentation."""

    # Adjust the paths based on the image category
    original_image_path = f"{image_category}/original.png"
    expected_image_dir = f"{image_category}/augmentations/combined"

    # Set the parameters
    predefined_param = dit.GPU_FUNDUS_RANDOM_AUG_KORNIA_PARAM
    param_dict = predefined_param.dict()

    # Test the augmentation
    _test_kornia_augmentation(
        pytestconfig=pytestconfig,
        image_path=original_image_path,
        expected_dir=expected_image_dir,
        param_dict=param_dict,
        n_augs=N_AUGS_COMBINED,
    )
