from typing import Dict, List, Optional, Tuple

import albumentations as A
import numpy as np

from ..utils import custom_albumentations as CA
from ..utils import improc
from ..utils import types as tp


class Transform:
    """Single image transform class."""

    def __init__(
        self,
        param: tp.TransformParam,
        target_shape: Optional[Tuple[int, int]] = None,
        normalize: Optional[Dict[str, Tuple[float, float, float]]] = None,
    ) -> None:
        """Initialize a transform.

        Args:
            param: The parameters for the transform.
            target_shape: The target shape for the transformed image.
            normalize: The normalization parameters (mean, std).
        """

        self._param = param
        self._target_shape = target_shape
        self._normalize = normalize

        self._geometric_albumentation: Optional[A.BasicTransform] = None
        self._photometric_albumentation: Optional[A.BasicTransform] = None
        self._post_albumentation: Optional[A.BasicTransform] = None

        self._set_geometric_transform()
        self._set_photometric_transform()
        self._set_post_transform()

    def _set_geometric_transform(self) -> None:
        """Set the geometric albumentation transform."""

        albumentations = [
            A.Affine(
                translate_percent={
                    "x": self._param.translate_x,
                    "y": self._param.translate_y,
                },
                rotate=self._param.rotate,
                shear={"x": self._param.shear_x, "y": self._param.shear_y},
                always_apply=True,
            ),
            A.HorizontalFlip(p=self._param.hflip),
            A.VerticalFlip(p=self._param.vflip),
        ]
        self._geometric_albumentation = A.Compose(albumentations)

    def _set_photometric_transform(self) -> None:
        """Set the photometric albumentation transform."""

        albumentations = [
            A.RandomBrightnessContrast(
                brightness_limit=[self._param.brightness, self._param.brightness],
                contrast_limit=[self._param.contrast, self._param.contrast],
                always_apply=True,
            ),
        ]
        self._photometric_albumentation = A.Compose(albumentations)

    def _set_post_transform(self) -> None:
        """Set the post albumentation transform."""

        albumentations: List[A.BasicTransform] = []
        if self._normalize:
            albumentations.append(A.Normalize(**self._normalize))
        if albumentations:
            self._post_albumentation = A.Compose(albumentations)

    def _pre_transform(self, image: np.ndarray) -> np.ndarray:
        """Pre-transform an image."""

        # Preprop (if required)
        if self._param.precrop:
            image = improc.center_crop_square(image, self._param.precrop)

        # Circle mask (if required)
        if self._param.circle_mask:
            image = improc.mask_center_circle(image)

        # Color space transfer (if required)
        if self._param.color_transfer:
            image = improc.color_transfer(image, self._param.color_transfer)

        return image

    def _geometric_transform(self, image: np.ndarray) -> np.ndarray:
        """Geometrically transform an image."""

        # If no target shape is given, use the original image shape
        target_shape = self._target_shape or image.shape[:2]

        # Compute the image shape to begin with
        height, width = improc.compute_aspect_preserving_shape(image.shape[:2], target_shape)
        width = int(width * self._param.scale * self._param.aspect**0.5)
        height = int(height * self._param.scale / self._param.aspect**0.5)

        # Create Albumentations transform
        transform = A.Compose(
            [
                A.Resize(width=width, height=height),
                self._geometric_albumentation,
                CA.CenterCropOrPad(img_shape=target_shape),
            ]
        )

        # Apply transform to image
        return transform(image=image)["image"]

    def _photometric_transform(self, image: np.ndarray) -> np.ndarray:
        """Photometrically transform an image."""

        # Apply transform to image
        return self._photometric_albumentation(image=image)["image"]

    def _post_transform(self, image: np.ndarray) -> np.ndarray:
        """Post-transform an image."""

        if self._post_albumentation:
            image = self._post_albumentation(image=image)["image"]

        return image

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Transform an image.

        Args:
            image: The image to transform.

        Returns:
            The transformed image.
        """

        image = self._pre_transform(image)
        image = self._geometric_transform(image)
        image = self._photometric_transform(image)
        image = self._post_transform(image)
        return image
