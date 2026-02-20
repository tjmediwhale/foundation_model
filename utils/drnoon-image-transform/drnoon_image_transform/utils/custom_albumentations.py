from typing import Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np

from . import improc


class CenterCropOrPad(A.ImageOnlyTransform):
    """Center crop or pad an image to the given size."""

    def __init__(
        self,
        img_shape: Tuple[int, int],
        pad_value: Union[int, float] = 0,
        always_apply: bool = False,
        p: float = 1.0,
    ) -> None:
        super().__init__(always_apply, p)

        if len(img_shape) != 2:
            raise ValueError("Image shape must have two values (height, width)")
        self.img_shape = img_shape
        self.pad_value = pad_value

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("img_shape", "pad_value")

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return improc.center_crop_or_pad(img, self.img_shape, self.pad_value)


class FundusContrastEnhancement(A.ImageOnlyTransform):
    """Contrast enhancement of fundus images.

    Reference:
        https://www.researchgate.net/figure/Contrast-enhancement-of-the-Kaggle-EyePACS-fundus-image-The-Gaussian-blur-technique-was_fig3_340563795
    """

    def __init__(
        self,
        img_shape: Optional[Tuple[int, int]] = None,
        always_apply=False,
        p=1.0,
    ) -> None:
        super().__init__(always_apply, p)

        if img_shape is not None:
            if len(img_shape) != 2:
                raise ValueError("Image shape must have two values (height, width)")
            self.circle_mask = improc.generate_center_circle_mask(img_shape)
        self.img_shape = img_shape

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("img_shape",)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        # Validate the image dtype
        if img.dtype != np.uint8:
            raise TypeError(f"Image dtype ({img.dtype}) must be np.uint8")

        if self.img_shape is None:
            # If image shape wasn't predefined, generate a circle mask based on the current image shape
            circle_mask = improc.generate_center_circle_mask(img.shape[:2])
            blur_sigma = min(img.shape[:2]) / 30
        else:
            # If image shape was predefined, check consistency with the shape of the current image
            if img.shape[:2] != self.img_shape:
                raise ValueError(
                    f"Image shape {img.shape[:2]} does not match the expected shape {self.img_shape}"
                )
            circle_mask = self.circle_mask
            blur_sigma = min(self.img_shape) / 30

        # Convert to float32 to avoid overflow
        img = img.astype(np.float32)

        # Enhance contrast by blurring and weighted addition
        blurred_img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
        contrasted_img = cv2.addWeighted(img, 4, blurred_img, -4, 128)

        # Apply circular mask: pixels outside the circle take a constant value
        contrasted_img[~circle_mask] = 128

        # Clip values to 8-bit range and convert back to uint8
        contrasted_img = contrasted_img.clip(0, 255).astype(np.uint8)
        return contrasted_img


class GaussianBlackout(A.ImageOnlyTransform):
    def __init__(
        self,
        kernel_size_range: Tuple[float, float] = (0.1, 1),
        alpha_range: Tuple[float, float] = (0.1, 1),
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super().__init__(always_apply, p)

        self.kernel_size_range = kernel_size_range
        self.alpha_range = alpha_range

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("kernel_size_range", "alpha_range")

    def _compute_elliptical_gaussian_kernel(self, img_h: int, img_w: int) -> np.ndarray:
        """Compute an elliptical Gaussian kernel.""" ""

        # Randomly sample kernel size and alpha values from the given ranges
        kernel_size = np.random.uniform(*self.kernel_size_range)

        # Select random center point for the Gaussian kernel
        cy = np.random.randint(0, img_h)
        cx = np.random.randint(0, img_w)

        # Create grids centered at the randomly selected point
        X, Y = np.meshgrid(np.arange(img_w) - cx, np.arange(img_h) - cy)

        # Randomly sample coefficients for the elliptical Gaussian kernel
        w_x = np.random.uniform(0.5, 1.5)
        w_xy = np.random.uniform(-0.5, 0.5)
        w_y = np.random.uniform(0.5, 1.5)

        # Compute squared distances from the center using elliptical equation
        elliptical_dist = w_x * X**2 + w_xy * X * Y + w_y * Y**2

        # Compute the Gaussian kernel values based on the elliptical distance
        gaussian_kernel = np.exp(-elliptical_dist * np.pi / (kernel_size * img_h * img_w))

        return gaussian_kernel

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Apply a soft elliptical blackout to an image."""

        # Compute the elliptical Gaussian kernel
        gaussian_kernel = self._compute_elliptical_gaussian_kernel(img.shape[0], img.shape[1])

        # Randomly sample alpha value from the given range
        alpha = np.random.uniform(*self.alpha_range)

        # Create a mask using the Gaussian kernel and alpha value
        mask = 1 - alpha * gaussian_kernel

        # If the image has multiple channels, repeat the mask for each channel
        if img.ndim == 3:
            mask = np.expand_dims(mask, axis=2)

        # Apply the mask on the image
        return np.clip(img * mask, 0, 255).astype(np.uint8)
