from typing import Dict, List, Optional, Tuple

import kornia as K
import kornia.augmentation as KA
import torch
import torchvision
from torch import nn

from ..utils import custom_kornia_augmentations as CKA
from ..utils import types as tp


class RandomAugmentationKornia(nn.Module):
    """Random augmentation module using kornia."""

    def __init__(
        self,
        param: tp.RandomAugParam,
        target_shape: Optional[Tuple[int, int]] = None,
        normalize: Optional[Dict[str, Tuple[float, float, float]]] = None,
    ) -> None:
        super().__init__()
        """Initialize RandomAugmentationKornia module.

        Args:
            param (tp.RandomAugParam): Random augmentation parameter.
            target_shape (Optional[Tuple[int, int]], optional): Target image shape. Defaults to None.
            normalize (Optional[Dict[str, Tuple[float, float, float]]], optional): Normalization parameters. Defaults to None.
        """

        self._param = param
        self._target_shape = target_shape
        self._normalize = normalize

        self._photometric_kornia: List[K.AugmentationBase2D] = []
        self._post_kornia: Optional[K.AugmentationBase2D] = None

        self._set_photometric_transform()
        self._set_post_transform()

    def _set_photometric_transform(self) -> None:
        """Set the photometric albumentation transforms."""

        photo_param = self._param.photometric

        kornias = [
            KA.ColorJiggle(**photo_param.color_jitter),
            KA.RandomPosterize(**photo_param.posterize),
            KA.RandomSolarize(**photo_param.solarize),
            KA.RandomSharpness(**photo_param.sharpen),
            KA.RandomEqualize(**photo_param.equalize),
            CKA.FundusContrastEnhancementKornia(
                img_shape=self._target_shape, **photo_param.fundus_contrast_enhancement
            ),
        ]

        kornias = [kor for kor in kornias if kor.p > 0.0]

        self._photometric_kornia = KA.container.ImageSequential(
            *kornias, same_on_batch=False, random_apply=True
        )

    def _set_post_transform(self) -> None:
        """Set the post kornia transform."""

        kornias: List[K.AugmentationBase2D] = []
        if self._normalize:
            kornias.append(KA.Normalize(**self._normalize))
        if self._param.postaug.coarse_dropout:
            kornias.append(CKA.CoarseDropoutKornia(**self._param.postaug.coarse_dropout))
        if kornias:
            self._post_kornia = nn.Sequential(*kornias)

    def _photometric_transform(self, image: torch.Tensor) -> torch.Tensor:
        """Photometrically transform an image."""
        # Apply transform to image
        return self._photometric_kornia(image)

    def _post_transform(self, image: torch.Tensor) -> torch.Tensor:
        """Post-transform an image."""

        if self._post_kornia:
            image = self._post_kornia(image)

        return image

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = self._photometric_transform(image)
        image = self._post_transform(image)
        return image
