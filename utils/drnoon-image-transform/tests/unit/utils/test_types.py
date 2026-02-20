import pytest

from drnoon_image_transform.utils import constants as ct
from drnoon_image_transform.utils.types import (
    RGBColor,
    RGBColorSpace,
    TransformParam,
    TTAParam,
)


class TestRGBColor:
    @pytest.mark.parametrize(
        "r, g, b",
        [
            (0, 0, 0),
            (255, 255, 255),
        ],
    )
    def test_rgb_color(self, r: float, g: float, b: float):
        color = RGBColor(r=r, g=g, b=b)
        assert (color.r, color.g, color.b) == (r, g, b)

    @pytest.mark.parametrize(
        "r, g, b",
        [
            (-1, 0, 0),
            (256, 255, 255),
        ],
    )
    def test_rgb_color_invalid(self, r: float, g: float, b: float):
        with pytest.raises(ValueError):
            RGBColor(r=r, g=g, b=b)


class TestRGBColorSpace:
    def test_rgb_color_space_default(self):
        cs = RGBColorSpace()
        assert (cs.mean.r, cs.mean.g, cs.mean.b) == ct.FUNDUS_RGB_MEAN
        assert (cs.std.r, cs.std.g, cs.std.b) == ct.FUNDUS_RGB_STD

    @pytest.mark.parametrize("mean", [(0, 0, 0), (255, 255, 255)])
    @pytest.mark.parametrize("std", [(128, 128, 128), (255, 255, 255)])
    def test_rgb_color(self, mean: tuple, std: tuple):
        cs = RGBColorSpace(
            mean={"r": mean[0], "g": mean[1], "b": mean[2]},
            std={"r": std[0], "g": std[1], "b": std[2]},
        )
        assert (cs.mean.r, cs.mean.g, cs.mean.b) == mean
        assert (cs.std.r, cs.std.g, cs.std.b) == std

        cs = RGBColorSpace(mean=mean, std=std)
        assert (cs.mean.r, cs.mean.g, cs.mean.b) == mean
        assert (cs.std.r, cs.std.g, cs.std.b) == std

        cs = RGBColorSpace(mean=list(mean), std=list(std))
        assert (cs.mean.r, cs.mean.g, cs.mean.b) == mean
        assert (cs.std.r, cs.std.g, cs.std.b) == std


class TestTransformParam:
    def test_transform_param_default(self):
        param = TransformParam()
        assert param.precrop is None
        assert param.color_transfer is None
        assert param.scale == 1.0
        assert param.aspect == 1.0
        assert param.rotate == 0.0
        assert param.translate_x == 0.0
        assert param.translate_y == 0.0
        assert param.shear_x == 0.0
        assert param.shear_y == 0.0
        assert param.hflip is False
        assert param.vflip is False
        assert param.brightness == 0.0
        assert param.contrast == 0.0

    @pytest.mark.parametrize(
        "param_dict",
        [
            ({"precrop": 0.5}),
            ({"color_transfer": RGBColorSpace()}),
            ({"scale": 0.5}),
            ({"aspect": 0.5}),
            ({"rotate": 45}),
            ({"translate_x": 0.5}),
            ({"translate_y": 0.5}),
            ({"shear_x": 45}),
            ({"shear_y": 45}),
            ({"hflip": True}),
            ({"vflip": True}),
            ({"brightness": 0.5}),
            ({"contrast": 0.5}),
            (
                {
                    "precrop": 0.5,
                    "color_transfer": RGBColorSpace(),
                    "scale": 0.5,
                    "aspect": 0.5,
                    "rotate": 45,
                    "translate_x": 0.5,
                    "translate_y": 0.5,
                    "shear_x": 45,
                    "shear_y": 45,
                    "hflip": True,
                    "vflip": True,
                    "brightness": 0.5,
                    "contrast": 0.5,
                }
            ),
        ],
    )
    def test_precrop(self, param_dict):
        param = TransformParam(**param_dict)
        for k, v in param_dict.items():
            assert getattr(param, k) == v

    @pytest.mark.parametrize(
        "param_dict",
        [
            ({"precrop": -0.1}),  # out of range
            ({"precrop": 1.1}),  # out of range
            ({"preccrop": 1.0}),  # invalid key
            # ...
        ],
    )
    def test_invalid_precrop(self, param_dict):
        with pytest.raises(ValueError):
            TransformParam(**param_dict)


class TestTTAParam:
    def test_tta_param_default(self):
        param = TTAParam()
        assert param.precrop == [None]
        assert param.color_transfer == [None]
        assert param.scale == [1.0]
        assert param.aspect == [1.0]
        assert param.rotate == [0.0]
        assert param.translate_x == [0.0]
        assert param.translate_y == [0.0]
        assert param.shear_x == [0.0]
        assert param.shear_y == [0.0]
        assert param.hflip == [False]
        assert param.vflip == [False]
        assert param.brightness == [0.0]
        assert param.contrast == [0.0]
