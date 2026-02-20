from functools import partial
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, confloat, conlist, conset, validator

from . import constants as ct

Fraction = confloat(gt=0, le=1)
Ratio = confloat(gt=0)
Angle = confloat(ge=-180, le=180)
Probability = confloat(ge=0, le=1)
# pydantic v2 호환: min_items/max_items -> min_length/max_length
# pydantic v2에서는 min_items/max_items가 min_length/max_length로 변경됨
import pydantic
if hasattr(pydantic, '__version__') and int(pydantic.__version__.split('.')[0]) >= 2:
    # pydantic v2
    Range = partial(conlist, min_length=2, max_length=2)
    Scope = partial(conset, min_length=1)
else:
    # pydantic v1
    Range = partial(conlist, min_items=2, max_items=2)
    Scope = partial(conset, min_items=1)
AlbuDict = Dict[str, Any]


def default(value: Any) -> Field:
    return Field(default_factory=lambda: value)


class StrictBaseModel(BaseModel):
    class Config:
        validate_assignment = True
        extra = "forbid"


class RGBColor(StrictBaseModel):
    r: confloat(ge=0, le=255)
    g: confloat(ge=0, le=255)
    b: confloat(ge=0, le=255)


class RGBColorSpace(StrictBaseModel):
    mean: RGBColor = Field(
        default_factory=lambda: RGBColor(
            r=ct.FUNDUS_RGB_MEAN[0],
            g=ct.FUNDUS_RGB_MEAN[1],
            b=ct.FUNDUS_RGB_MEAN[2],
        )
    )
    std: RGBColor = Field(
        default_factory=lambda: RGBColor(
            r=ct.FUNDUS_RGB_STD[0],
            g=ct.FUNDUS_RGB_STD[1],
            b=ct.FUNDUS_RGB_STD[2],
        )
    )

    @validator("mean", "std", pre=True)
    def convert_to_rgb_color(cls, v):
        if isinstance(v, tuple) or isinstance(v, list):
            if len(v) != 3:
                raise ValueError("RGBColorSpace mean/std must be 3-channel")
            return RGBColor(r=v[0], g=v[1], b=v[2])
        return v

    @property
    def mean_tuple(self):
        return (self.mean.r, self.mean.g, self.mean.b)

    @property
    def std_tuple(self):
        return (self.std.r, self.std.g, self.std.b)


class TransformParam(StrictBaseModel):
    precrop: Optional[Fraction] = None  # center precrop ratio to the original (e.g. 0.5, 1.0)
    circle_mask: bool = False  # whether to apply a circular mask to the center {True, False}
    color_transfer: Optional[RGBColorSpace] = None  # color space to transfer to
    # Geometric transforms
    scale: Ratio = 1.0  # relative scale to the original (e.g. 0.5, 1, 2)
    aspect: Ratio = 1.0  # relative aspect ratio (width/height) to the original (e.g. 0.7, 1, 1.3)
    rotate: Angle = 0.0  # rotation angle in degree (e.g. -45, 0, 45)
    translate_x: float = 0.0  # relative translation to width (e.g. -0.3, 0, 0.3)
    translate_y: float = 0.0  # relative translation to height (e.g. -0.3, 0, 0.3)
    shear_x: Angle = 0.0  # shearing angle in degree (e.g. -45, 0, 45)
    shear_y: Angle = 0.0  # shearing angle in degree (e.g. -45, 0, 45)
    hflip: bool = False  # whether to flip horizontally {True, False}
    vflip: bool = False  # whether to flip vertically {True, False}
    # Photometric transforms
    brightness: float = 0.0  # relative brightness change to the original (e.g. -0.3, 0, 0.3)
    contrast: float = 0.0  # relative contrast change to the original (e.g. -0.3, 0, 0.3)


class TTAParam(StrictBaseModel):
    precrop: List[Optional[Fraction]] = default([None])
    circle_mask: List[bool] = default([False])
    color_transfer: List[Optional[RGBColorSpace]] = default([None])
    # Geometric transforms
    scale: List[Ratio] = default([1.0])
    aspect: List[Ratio] = default([1.0])
    rotate: List[Angle] = default([0.0])
    translate_x: List[float] = default([0.0])
    translate_y: List[float] = default([0.0])
    shear_x: List[Angle] = default([0.0])
    shear_y: List[Angle] = default([0.0])
    hflip: List[bool] = default([False])
    vflip: List[bool] = default([False])
    # Photometric transforms
    brightness: List[float] = default([0.0])
    contrast: List[float] = default([0.0])


class GeoAugParam(StrictBaseModel):
    """Random geometric augmentation parameters."""

    scale: Range(Ratio) = default([1.0, 1.0])
    aspect: Range(Ratio) = default([1.0, 1.0])
    rotate: Range(Angle) = default([0, 0])
    translate_x: Range(float) = default([0, 0])
    translate_y: Range(float) = default([0, 0])
    shear_x: Range(Angle) = default([0, 0])
    shear_y: Range(Angle) = default([0, 0])
    hflip: Probability = 0
    vflip: Probability = 0


class PhotoAugParam(StrictBaseModel):
    """Random photometric augmentation parameters.

    The transforms will run in a random order.
    """

    # Color
    color_jitter: AlbuDict = default({"p": 0})
    random_brightness_contrast: AlbuDict = default({"p": 0})
    random_gamma: AlbuDict = default({"p": 0})
    # Degradation
    blur: AlbuDict = default({"p": 0})
    gaussian_blur: AlbuDict = default({"p": 0})
    median_blur: AlbuDict = default({"p": 0})
    motion_blur: AlbuDict = default({"p": 0})
    zoom_blur: AlbuDict = default({"p": 0})
    defocus: AlbuDict = default({"p": 0})
    downscale: AlbuDict = default({"p": 0})
    image_compression: AlbuDict = default({"p": 0})
    posterize: AlbuDict = default({"p": 0})
    solarize: AlbuDict = default({"p": 0})
    # Enhancement
    sharpen: AlbuDict = default({"p": 0})
    equalize: AlbuDict = default({"p": 0})
    clahe: AlbuDict = default({"p": 0})
    # Noise
    gauss_noise: AlbuDict = default({"p": 0})
    iso_noise: AlbuDict = default({"p": 0})
    multiplicative_noise: AlbuDict = default({"p": 0})
    # Custom albumentations
    gaussian_blackout: AlbuDict = default({"p": 0})
    fundus_contrast_enhancement: AlbuDict = default({"p": 0})


class PreAugParam(StrictBaseModel):
    precrop: Optional[Range(Fraction)] = None
    circle_mask: Probability = 0


class PostAugParam(StrictBaseModel):
    coarse_dropout: Optional[AlbuDict] = None


class RandomAugParam(StrictBaseModel):
    """Random augmentation parameters."""

    preaug: PreAugParam = Field(default_factory=PreAugParam)
    geometric: GeoAugParam = Field(default_factory=GeoAugParam)
    photometric: PhotoAugParam = Field(default_factory=PhotoAugParam)
    postaug: PostAugParam = Field(default_factory=PostAugParam)
