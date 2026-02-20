from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..transform import Transform
from ..utils import types as tp


class TTA:
    """Test Time Augmentation (TTA) class."""

    def __init__(
        self,
        param: tp.TTAParam,
        target_shape: Optional[Tuple[int, int]] = None,
        normalize: Optional[Dict[str, Tuple[float, float, float]]] = None,
    ) -> None:
        """Initialize a TTA.

        Args:
            param: The TTA parameters to define the transforms.
            target_shape: The target shape for the transformed image.
            normalize: The normalization parameters (mean, std).
        """

        self._tta_param = param
        self._target_shape = target_shape
        self._normalize = normalize

        self._transforms: List[Transform] = self._generate_transforms()

    @property
    def transforms(self) -> List[Transform]:
        """The list of image transforms for the TTA."""

        return self._transforms

    def _generate_transforms(self) -> List[Transform]:
        """Generate all possible transforms from the TTA parameters.

        Returns:
            A list of transforms for all possible combinations of the TTA parameters.
        """

        tta_param_dict = self._tta_param.dict()
        transforms: List[Transform] = []
        for combination in product(*tta_param_dict.values()):
            transform_param = tp.TransformParam(**dict(zip(tta_param_dict.keys(), combination)))
            transform = Transform(
                param=transform_param,
                target_shape=self._target_shape,
                normalize=self._normalize,
            )
            transforms.append(transform)
        return transforms

    def __call__(self, image: np.ndarray) -> List[np.ndarray]:
        """Apply TTA to an image.

        Args:
            image: The image to transform.

        Returns:
            A list of transformed images.
        """

        return [transform(image) for transform in self._transforms]
