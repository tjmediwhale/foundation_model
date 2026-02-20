from typing import Any, Dict, Optional, Tuple, Union

import kornia.augmentation as KA
import numpy as np
import torch
import torchvision

from . import improc


class FundusContrastEnhancementKornia(KA.IntensityAugmentationBase2D):
    def __init__(
        self,
        img_shape: Optional[Tuple[int, int]] = None,
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.img_shape = img_shape
        if img_shape is not None:
            if len(img_shape) != 2:
                raise ValueError("Image shape must have two values (height, width)")
            self.circle_mask = torch.from_numpy(improc.generate_center_circle_mask(img_shape)).to(
                torch.float32
            )
        self.img_shape = img_shape

    def apply_transform(
        self,
        img: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.img_shape is None:
            # If image shape wasn't predefined, generate a circle mask based on the current image shape
            circle_mask = torch.from_numpy(improc.generate_center_circle_mask(img.shape[2:])).to(
                torch.float32
            )
            blur_sigma = min(img.shape[2:]) / 30
        else:
            # If image shape was predefined, check consistency with the shape of the current image
            if img.shape[2:] != self.img_shape:
                raise ValueError(
                    f"Image shape {img.shape[2:]} does not match the expected shape {self.img_shape}"
                )
            if img.device != self.circle_mask.device:
                self.circle_mask = self.circle_mask.to(img.device)

            circle_mask = self.circle_mask
            blur_sigma = min(self.img_shape) / 30

        # Convert to float32 to avoid overflow
        img = img.to(torch.float32)

        # Apply Gaussian blur
        blur_img = torchvision.transforms.functional.gaussian_blur(
            img, kernel_size=int(blur_sigma) * 2 + 1, sigma=blur_sigma
        )

        # Increase contrast and merge
        contrasted_img = 4 * img - 4 * blur_img + 0.5  # 0.5 is for the 128 offset
        contrasted_img = contrasted_img * circle_mask + 0.5 * (
            1 - circle_mask
        )  # Apply circular mask
        contrasted_img.clamp_(0, 1)  # Clamp to valid image range

        return contrasted_img


class CoarseDropoutKornia(KA.IntensityAugmentationBase2D):
    def __init__(
        self,
        max_holes: int = 8,
        max_height: float = 0.25,
        max_width: float = 0.25,
        min_holes: int = 1,
        min_height: float = 0.01,
        min_width: float = 0.01,
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width
        self.min_holes = min_holes
        self.min_height = min_height
        self.min_width = min_width

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, C, H, W = input.shape
        num_holes = np.random.randint(self.min_holes, self.max_holes + 1)
        holes = []
        for _ in range(num_holes):
            hole_height = np.random.uniform(self.min_height, self.max_height) * H
            hole_width = np.random.uniform(self.min_width, self.max_width) * W

            y1 = np.random.uniform(0, H - hole_height)
            x1 = np.random.uniform(0, W - hole_width)
            y2 = y1 + hole_height
            x2 = x1 + hole_width

            input[:, :, int(y1) : int(y2), int(x1) : int(x2)] = 0

        return input
