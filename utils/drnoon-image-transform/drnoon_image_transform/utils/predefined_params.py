from . import types as tp

WIDE_TRANSFORM_PARAM = tp.TransformParam(
    precrop=0.35,
    circle_mask=True,
)

FUNDUS_TTA_PARAM = tp.TTAParam(
    scale=[0.7, 1, 1.3],
    rotate=[-60, -30, 0, 30, 60],
    hflip=[True, False],
)

WIDE_FUNDUS_TTA_PARAM = tp.TTAParam(
    scale=[0.7, 1, 1.3],
    rotate=[-60, -30, 0, 30, 60],
    hflip=[True, False],
    precrop=[0.35],
    circle_mask=[True],
)

FUNDUS_RANDOM_AUG_PARAM = tp.RandomAugParam(
    geometric=tp.GeoAugParam(
        scale=[0.5, 1.5],
        aspect=[0.5, 1.5],
        rotate=[-75, 75],
        translate_x=[-0.3, 0.3],
        translate_y=[-0.3, 0.3],
        shear_x=[-30, 30],
        shear_y=[-30, 30],
        hflip=0.5,
        vflip=0.5,
    ),
    photometric=tp.PhotoAugParam(
        color_jitter={
            "brightness": 0.3,
            "contrast": 0.3,
            "saturation": 0.3,
            "hue": 0.3,
            "p": 0.9,
        },
        gaussian_blackout={"p": 0.1},
        random_brightness_contrast={"p": 0.05},
        random_gamma={"p": 0.05},
        blur={"p": 0.05},
        gaussian_blur={"p": 0.05},
        median_blur={"p": 0.05},
        motion_blur={"p": 0.05},
        zoom_blur={"max_factor": 1.21, "p": 0.05},
        defocus={"radius": 7, "p": 0.05},
        downscale={"scale_min": 0.25, "scale_max": 0.99, "p": 0.05},
        image_compression={"quality_lower": 25, "quality_upper": 99, "p": 0.05},
        posterize={"num_bits": [4, 8], "p": 0.05},
        solarize={"p": 0.05, "threshold": [128 - int(0.2 * 256), 128 + int(0.2 * 256)]},
        sharpen={"p": 0.05},
        equalize={"p": 0.05},
        clahe={"p": 0.05},
        gauss_noise={"p": 0.05},
        iso_noise={"p": 0.05},
        multiplicative_noise={"p": 0.05},
        fundus_contrast_enhancement={"p": 0.05},
    ),
    postaug=tp.PostAugParam(
        coarse_dropout={
            "max_holes": 16,
            "max_height": 0.25,
            "max_width": 0.25,
            "min_holes": 1,
            "min_height": 0.01,
            "min_width": 0.01,
            "p": 0.3,
        },
    ),
)

CPU_FUNDUS_RANDOM_AUG_PRE_KORNIA_PARAM = tp.RandomAugParam(
    geometric=tp.GeoAugParam(
        scale=[0.5, 1.5],
        aspect=[0.5, 1.5],
        rotate=[-75, 75],
        translate_x=[-0.3, 0.3],
        translate_y=[-0.3, 0.3],
        shear_x=[-30, 30],
        shear_y=[-30, 30],
        hflip=0.5,
        vflip=0.5,
    ),
    photometric=tp.PhotoAugParam(
        color_jitter={
            "brightness": 0.0,
            "contrast": 0.0,
            "saturation": 0.0,
            "hue": 0.0,
            "p": 0.0,
        },
        gaussian_blackout={"p": 0.1},
        random_brightness_contrast={"p": 0.05},
        random_gamma={"p": 0.05},
        blur={"p": 0.05},
        gaussian_blur={"p": 0.05},
        median_blur={"p": 0.05},
        motion_blur={"p": 0.05},
        zoom_blur={"max_factor": 1.21, "p": 0.05},
        defocus={"radius": 7, "p": 0.05},
        downscale={"scale_min": 0.25, "scale_max": 0.99, "p": 0.05},
        image_compression={"quality_lower": 25, "quality_upper": 99, "p": 0.05},
        posterize={"num_bits": [4, 8], "p": 0.00},
        solarize={"p": 0.00},
        sharpen={"p": 0.00},
        equalize={"p": 0.00},
        clahe={"p": 0.05},
        gauss_noise={"p": 0.05},
        iso_noise={"p": 0.05},
        multiplicative_noise={"p": 0.05},
        fundus_contrast_enhancement={"p": 0.00},
    ),
    postaug=tp.PostAugParam(
        coarse_dropout=None,
    ),
)

GPU_FUNDUS_RANDOM_AUG_KORNIA_PARAM = tp.RandomAugParam(
    photometric=tp.PhotoAugParam(
        color_jitter={
            "brightness": 0.3,
            "contrast": 0.3,
            "saturation": 0.3,
            "hue": 0.3,
            "p": 0.9,
        },
        posterize={"bits": [4, 8], "p": 0.05},
        solarize={"p": 0.05},
        sharpen={"p": 0.05},
        equalize={"p": 0.05},
        fundus_contrast_enhancement={"p": 0.05},
    ),
    postaug=tp.PostAugParam(
        coarse_dropout={
            "max_holes": 16,
            "max_height": 0.25,
            "max_width": 0.25,
            "min_holes": 1,
            "min_height": 0.01,
            "min_width": 0.01,
            "p": 0.3,
        },
    ),
)

CPU_WIDE_RANDOM_AUG_PRE_KORNIA_PARAM = tp.RandomAugParam(
    preaug=tp.PreAugParam(
        precrop=[0.2, 0.5],
        circle_mask=0.1,
    ),
    geometric=CPU_FUNDUS_RANDOM_AUG_PRE_KORNIA_PARAM.geometric,
    photometric=CPU_FUNDUS_RANDOM_AUG_PRE_KORNIA_PARAM.photometric,
    postaug=CPU_FUNDUS_RANDOM_AUG_PRE_KORNIA_PARAM.postaug,
)

WIDE_RANDOM_AUG_PARAM = tp.RandomAugParam(
    preaug=tp.PreAugParam(
        precrop=[0.2, 0.5],
        circle_mask=0.1,
    ),
    geometric=FUNDUS_RANDOM_AUG_PARAM.geometric,
    photometric=FUNDUS_RANDOM_AUG_PARAM.photometric,
    postaug=FUNDUS_RANDOM_AUG_PARAM.postaug,
)
