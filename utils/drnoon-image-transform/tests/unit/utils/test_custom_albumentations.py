import numpy as np
import pytest

from drnoon_image_transform.utils import custom_albumentations as CA

IMAGE_SHAPES = [(100, 100), (100, 200), (200, 100)]


class TestFundusContrastEnhancement:
    @pytest.mark.parametrize("image_shape", IMAGE_SHAPES)
    def test_initialization(self, image_shape):
        """Test the basic initialization of the transform class."""

        transform = CA.FundusContrastEnhancement(image_shape)
        assert transform.img_shape == image_shape
        assert transform.circle_mask.shape == image_shape

    @pytest.mark.parametrize("random_image", IMAGE_SHAPES, indirect=True)
    def test_apply_type_mismatch(self, random_image: np.ndarray):
        """Test the apply method when there is a type mismatch."""

        transform = CA.FundusContrastEnhancement(random_image.shape[:2])
        image = random_image.astype(np.float32)
        with pytest.raises(TypeError):
            transform.apply(image)

    @pytest.mark.parametrize("random_image", IMAGE_SHAPES, indirect=True)
    def test_apply_shape_mismatch(self, random_image: np.ndarray):
        """Test the apply method when there is a shape mismatch."""

        transform_shape = (random_image.shape[0] + 1, random_image.shape[1])
        transform = CA.FundusContrastEnhancement(transform_shape)
        with pytest.raises(ValueError):
            transform.apply(random_image)

    @pytest.mark.parametrize("random_image", IMAGE_SHAPES, indirect=True)
    def test_apply(self, random_image: np.ndarray):
        """Test the apply method under normal conditions and verify the outcome."""

        transform = CA.FundusContrastEnhancement(random_image.shape[:2])
        transformed = transform.apply(random_image)

        assert transformed.shape == random_image.shape
        assert transformed.dtype == np.uint8
