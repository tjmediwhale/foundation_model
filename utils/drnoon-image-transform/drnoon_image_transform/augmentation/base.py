from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np

from ..utils import custom_albumentations as CA
from ..utils import improc
from ..utils import types as tp


class RandomAugmentation:
    """Random image augmentation class."""

    def __init__(
        self,
        param: tp.RandomAugParam,
        target_shape: Optional[Tuple[int, int]] = None,
        normalize: Optional[Dict[str, Tuple[float, float, float]]] = None,
    ) -> None:
        """Initialize a random augmentation.

        Args:
            param: The parameters for the random augmentation.
            target_shape: The target shape for the transformed image.
            normalize: The normalization parameters (mean, std).
        """

        self._param = param
        self._target_shape = target_shape
        self._normalize = normalize

        self._geometric_albumentation: Optional[A.BasicTransform] = None
        self._photometric_albumentations: List[A.BasicTransform] = []
        self._post_albumentation: Optional[A.BasicTransform] = None

        self._set_geometric_transform()
        self._set_photometric_transform()
        self._set_post_transform()

    def _set_geometric_transform(self) -> None:
        """Set the geometric albumentation transform."""

        geo_param = self._param.geometric

        self._geometric_albumentation = A.Compose(
            [
                A.Affine(
                    translate_percent={
                        "x": geo_param.translate_x,
                        "y": geo_param.translate_y,
                    },
                    rotate=geo_param.rotate,
                    shear={
                        "x": geo_param.shear_x,
                        "y": geo_param.shear_y,
                    },
                    p=1.0,
                ),
                A.HorizontalFlip(p=geo_param.hflip),
                A.VerticalFlip(p=geo_param.vflip),
            ]
        )

    def _set_photometric_transform(self) -> None:
        """Set the photometric albumentation transforms."""

        photo_param = self._param.photometric

        albumentations = [
            A.ColorJitter(**photo_param.color_jitter),
            A.RandomBrightnessContrast(**photo_param.random_brightness_contrast),
            A.RandomGamma(**photo_param.random_gamma),
            A.Blur(**photo_param.blur),
            A.GaussianBlur(**photo_param.gaussian_blur),
            A.MedianBlur(**photo_param.median_blur),
            A.MotionBlur(**photo_param.motion_blur),
            A.ZoomBlur(**photo_param.zoom_blur),
            A.Defocus(**photo_param.defocus),
            A.Downscale(interpolation=cv2.INTER_NEAREST, **photo_param.downscale),
            A.ImageCompression(**photo_param.image_compression),
            A.Posterize(**photo_param.posterize),
            A.Solarize(**photo_param.solarize),
            A.Sharpen(**photo_param.sharpen),
            A.Equalize(**photo_param.equalize),
            A.CLAHE(**photo_param.clahe),
            A.GaussNoise(**photo_param.gauss_noise),
            A.ISONoise(**photo_param.iso_noise),
            A.MultiplicativeNoise(**photo_param.multiplicative_noise),
            CA.GaussianBlackout(**photo_param.gaussian_blackout),
            CA.FundusContrastEnhancement(
                img_shape=self._target_shape, **photo_param.fundus_contrast_enhancement
            ),
        ]
        self._photometric_albumentations = [albu for albu in albumentations if albu.p > 0.0]

    def _set_post_transform(self) -> None:
        """Set the post albumentation transform."""

        albumentations: List[A.BasicTransform] = []
        if self._normalize:
            albumentations.append(A.Normalize(**self._normalize))
        if self._param.postaug.coarse_dropout:
            albumentations.append(A.CoarseDropout(**self._param.postaug.coarse_dropout))
        if albumentations:
            self._post_albumentation = A.Compose(albumentations)

    def _pre_transform(self, image: np.ndarray) -> np.ndarray:
        """Pre-transform an image."""

        pre_param = self._param.preaug

        # Precrop (if required)
        if pre_param.precrop is not None:
            # Use a random ratio from the given range
            ratio = np.random.uniform(*pre_param.precrop)
            image = improc.center_crop_square(image, ratio)

        # Circle mask (if required)
        if pre_param.circle_mask > 0:
            # Use the given value as the probability to apply circle mask
            if np.random.rand() < pre_param.circle_mask:
                image = improc.mask_center_circle(image)

        return image

    def _geometric_transform(self, image: np.ndarray) -> np.ndarray:
        """Geometrically transform an image."""

        # If no target shape is given, use the original image shape
        target_shape = self._target_shape or image.shape[:2]

        # Select random scale and aspect ratio from the given range
        scale = np.random.uniform(*self._param.geometric.scale)
        aspect = np.random.uniform(*self._param.geometric.aspect)

        # Compute the image shape to begin with
        height, width = improc.compute_aspect_preserving_shape(image.shape[:2], target_shape)
        width = int(width * scale * aspect**0.5)
        height = int(height * scale / aspect**0.5)

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

        # Randomly order the photometric albumentations
        albumentations = np.random.permutation(self._photometric_albumentations)

        # Create Albumentations transform
        transform = A.Compose(albumentations)

        # Apply transform to image
        return transform(image=image)["image"]

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
