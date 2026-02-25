"""
foundation_model utils/preprocessing.py
drnoon-image-transform 기반 Fundus 이미지 전처리 (retinal crop).
모든 augmentation 전에 적용하여 retinal 부분만 crop.
"""
from typing import Optional

import numpy as np
from PIL import Image


def fundus_preprocess_drnoon(
    pil_image: Image.Image,
    precrop: Optional[float] = 0.4,
    circle_mask: bool = True,
) -> Image.Image:
    """
    drnoon-image-transform으로 retinal 영역만 crop.
    precrop: center crop 비율 (0 < ratio <= 1). None이면 precrop 스킵.
    circle_mask: True면 원형 마스크 적용 (retinal 원만 남김).

    drnoon-ml 및 drnoon-image-transform 참고.
    """
    try:
        import sys
        from pathlib import Path
        # foundation_model 프로젝트 루트 기준 drnoon-image-transform 경로
        _this = Path(__file__).resolve()
        _dit_path = _this.parent / "drnoon-image-transform"
        if _dit_path.exists() and str(_dit_path) not in sys.path:
            sys.path.insert(0, str(_dit_path))

        from drnoon_image_transform.utils import improc
    except ImportError:
        # drnoon-image-transform 미설치 시 원본 반환
        return pil_image

    img_np = np.array(pil_image)
    if len(img_np.shape) == 2:
        img_np = np.stack([img_np] * 3, axis=-1)

    # _pre_transform과 동일: precrop -> circle_mask
    if precrop is not None and 0 < precrop <= 1:
        img_np = improc.center_crop_square(img_np, precrop)
    if circle_mask:
        img_np = improc.mask_center_circle(img_np)

    return Image.fromarray(img_np.astype(np.uint8))


def get_fundus_preprocess_fn(
    precrop: Optional[float] = 0.4,
    circle_mask: bool = True,
):
    """전처리 함수 반환 (config에서 precrop, circle_mask 주입용)."""
    def fn(pil_image: Image.Image) -> Image.Image:
        return fundus_preprocess_drnoon(pil_image, precrop=precrop, circle_mask=circle_mask)
    return fn
