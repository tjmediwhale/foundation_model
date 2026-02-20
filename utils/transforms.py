"""
foundation_model utils/transforms.py
CVD_update/scripts/train.sh 전처리/증강 로직을 동일 동작으로 재구성.
train/val/test 모두 동일 transforms 제공 (augment_val_test=True 강제).
"""
from typing import List, Optional, Union
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


# gs:// -> /nas/mediwhale_processed_data/ 치환은 data.py에서 수행

def _preprocessing_base(pil_image: Image.Image) -> Image.Image:
    """최소 외접원 크롭 (CVD_update data_loader와 동일)."""
    import cv2
    original_image = np.array(pil_image)
    image = (original_image[:, :, 0] + original_image[:, :, 1]) // 2
    image = cv2.GaussianBlur(image, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    hist = np.squeeze(cv2.calcHist([image], [0], None, [15], [0, 15]))
    total_pixels = np.sum(hist)
    current_pixels = float(total_pixels)
    log_hist = np.log(hist + 1)
    threshold = 2
    for threshold in range(2, 14):
        if (
            log_hist[threshold] < log_hist[threshold - 1]
            and log_hist[threshold] < log_hist[threshold + 1]
        ):
            break
        current_pixels -= float(hist[threshold])
        if current_pixels < total_pixels * 0.5:
            break
    mask = np.zeros_like(image)
    mask[:, :] = 255
    mask[image[:, :] <= threshold] = 0
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return pil_image
    cnt = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    mask = np.zeros_like(original_image)
    cv2.circle(mask, center, radius, (255, 255, 255), -1)
    masked_image = cv2.bitwise_and(original_image, mask)
    padded_image = cv2.copyMakeBorder(
        masked_image, radius, radius, radius, radius, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    cx, cy, r = int(x + radius), int(y + radius), int(radius)
    cropped_image = padded_image[cy - r : cy + r, cx - r : cx + r]
    return Image.fromarray(cropped_image.astype("uint8"))


def _preprocessing_wide_fundus(
    pil_image: Image.Image,
    precrop: float = 0.4,
    circle_mask: bool = True,
) -> Image.Image:
    """Wide Fundus 전처리 (drnoon_image_transform 사용, 없으면 원본 반환)."""
    try:
        import drnoon_image_transform as dit
        param = dit.TransformParam(precrop=precrop, circle_mask=circle_mask)
        transform = dit.Transform(param=param)
        return Image.fromarray(transform(np.array(pil_image)))
    except ImportError:
        return pil_image
    except Exception:
        return pil_image


def build_fundus_transform(
    image_size: int = 224,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    use_advanced_transforms: bool = True,
    use_preprocessing_base: bool = True,
    use_wide_fundus: bool = True,
    wide_fundus_precrop: float = 0.4,
    wide_fundus_circle_mask: bool = True,
    augment_val_test: bool = True,
    camera_type_column: Optional[str] = None,
) -> callable:
    """
    train.sh와 동일 동작의 transform 생성.
    augment_val_test=True이면 train/val/test 모두 동일 증강 적용.
    """
    def _apply_preprocessing(pil_image: Image.Image, is_wide: bool = False) -> Image.Image:
        if is_wide and use_wide_fundus:
            return _preprocessing_wide_fundus(
                pil_image,
                precrop=wide_fundus_precrop,
                circle_mask=wide_fundus_circle_mask,
            )
        if use_preprocessing_base and not is_wide:
            return _preprocessing_base(pil_image)
        return pil_image

    if use_advanced_transforms and (augment_val_test or True):
        albu_transform = A.Compose([
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0), ratio=(0.8, 1.2)),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=15, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.4),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
            A.Blur(blur_limit=3, p=0.2),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        albu_transform = A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    def transform_fn(pil_image: Image.Image, is_wide: bool = False) -> torch.Tensor:
        pil_image = _apply_preprocessing(pil_image, is_wide)
        img_np = np.array(pil_image)
        if len(img_np.shape) == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        out = albu_transform(image=img_np)
        return out["image"]

    return transform_fn


def get_fundus_transform_for_retfound(
    image_size: int = 224,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    augment_val_test: bool = True,
    **kwargs,
):
    """RETFound ImageFolder용 transform (torchvision Compose 형태)."""
    tf = build_fundus_transform(
        image_size=image_size,
        mean=mean,
        std=std,
        augment_val_test=augment_val_test,
        **kwargs,
    )

    def _wrap(img: Image.Image) -> torch.Tensor:
        return tf(img, is_wide=False)

    return _wrap
