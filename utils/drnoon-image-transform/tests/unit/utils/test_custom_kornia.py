import numpy as np
import pytest

CKA = pytest.importorskip("drnoon_image_transform.utils.custom_kornia_augmentations")

from tests.torch_helpers import (
    convert_np_array_to_torch_tensor,
    convert_torch_tensor_to_np_array,
)

IMAGE_SHAPES = [(100, 100), (100, 200), (200, 100)]


class TestFundusContrastEnhancementKornia:
    @pytest.mark.parametrize("image_shape", IMAGE_SHAPES)
    def test_initialization(self, image_shape):
        """Test the basic initialization of the transform class."""

        transform = CKA.FundusContrastEnhancementKornia(image_shape)
        assert transform.img_shape == image_shape
        assert transform.circle_mask.shape == image_shape

    @pytest.mark.parametrize("random_image", IMAGE_SHAPES, indirect=True)
    def test_apply_shape_mismatch(self, random_image: np.ndarray):
        """Test the apply method when there is a shape mismatch."""

        transform_shape = (random_image.shape[0] + 1, random_image.shape[1])
        transform = CKA.FundusContrastEnhancementKornia(transform_shape)
        random_image = convert_np_array_to_torch_tensor(random_image)
        with pytest.raises(ValueError):
            transform(random_image)

    @pytest.mark.parametrize("random_image", IMAGE_SHAPES, indirect=True)
    def test_apply(self, random_image: np.ndarray):
        """Test the apply method under normal conditions and verify the outcome."""

        transform = CKA.FundusContrastEnhancementKornia(random_image.shape[:2])
        random_image_shape = random_image.shape

        random_image = convert_np_array_to_torch_tensor(random_image)
        transformed = transform(random_image)
        transformed = convert_torch_tensor_to_np_array(transformed)

        assert transformed.shape == random_image_shape
        assert transformed.dtype == np.uint8
