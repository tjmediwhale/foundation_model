from typing import Tuple, Union

import cv2
import numpy as np

from .types import RGBColorSpace


def compute_aspect_preserving_shape(
    original_shape: Tuple[int, int],
    target_shape: Tuple[int, int],
) -> Tuple[int, int]:
    """Compute the new shape for resizing the original image, such that the aspect ratio is preserved and the target
    shape fits tightly within the new shape.

    Args:
        original_shape: The original image shape (height, width).
        target_shape: The target image shape (height, width).

    Returns:
        The new image shape (height, width).
    """

    original_height, original_width = original_shape
    target_height, target_width = target_shape

    # Calculate the aspect ratios
    original_aspect_ratio = original_height / original_width
    target_aspect_ratio = target_height / target_width

    # Determine which dimension to constrain
    if original_aspect_ratio > target_aspect_ratio:
        # Constrain by width
        new_width = target_width
        new_height = int(target_width * original_aspect_ratio)
    else:
        # Constrain by height
        new_height = target_height
        new_width = int(target_height / original_aspect_ratio)

    return (new_height, new_width)


def center_crop_or_pad(
    image: np.ndarray,
    target_shape: Tuple[int, int],
    pad_value: Union[int, float] = 0,
) -> np.ndarray:
    """Center crop or pad the given image to the target shape.

    Args:
        image: The original image with shape (height, width, channels).
        target_shape: The target shape (height, width).
        pad_value: The value to be used for padding.

    Returns:
        The cropped or padded image with target shape.
    """

    # Get the original and target dimensions
    original_height, original_width = image.shape[:2]
    target_height, target_width = target_shape

    # Calculate cropping dimensions
    crop_x1 = max((original_width - target_width) // 2, 0)
    crop_y1 = max((original_height - target_height) // 2, 0)
    crop_x2 = min(crop_x1 + target_width, original_width)
    crop_y2 = min(crop_y1 + target_height, original_height)

    # Crop the image
    cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]

    # Calculate padding dimensions
    pad_top = max((target_height - (crop_y2 - crop_y1)) // 2, 0)
    pad_bottom = max(target_height - (crop_y2 - crop_y1) - pad_top, 0)
    pad_left = max((target_width - (crop_x2 - crop_x1)) // 2, 0)
    pad_right = max(target_width - (crop_x2 - crop_x1) - pad_left, 0)

    # Pad the image
    cropped_or_padded = np.pad(
        cropped,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=pad_value,
    )
    return cropped_or_padded


def center_crop_square(image: np.ndarray, ratio: float) -> np.ndarray:
    """Center crop the given image into a square with the given ratio.

    Args:
        image: The original image with shape (height, width [, channels]).
        ratio: The ratio to the shorter side of the original image to determine the side of the square (0 < ratio <= 1).

    Returns:
        The cropped square image.
    """

    if ratio <= 0 or ratio > 1:
        raise ValueError(f"Invalid ratio: {ratio} (must be 0 < ratio <= 1)")

    # Get the original shape
    original_height, original_width = image.shape[:2]

    # Determine the side of the square based on the shorter dimension
    square_side = int(min(original_height, original_width) * ratio)

    # Calculate the coordinates of the top-left corner of the cropping area
    x1 = (original_width - square_side) // 2
    y1 = (original_height - square_side) // 2

    # Calculate the coordinates of the bottom-right corner
    x2 = x1 + square_side
    y2 = y1 + square_side

    # Crop the image to create a square
    cropped = image[y1:y2, x1:x2]

    return cropped


def generate_center_circle_mask(image_shape: Tuple[int, int]) -> np.ndarray:
    """Generate a circular mask with the given image shape.

    Args:
        image_shape: The mask shape (height, width).

    Returns:
        The circular mask with the given image shape.
    """

    # The image height and width
    height, width = image_shape

    # Create a mask image initialized to zeros
    mask = np.zeros((height, width), dtype=np.uint8)

    # Determine the center and radius of the circle
    center = (width // 2, height // 2)
    radius = min(center)

    # Draw a white circle at the center of the mask.
    cv2.circle(mask, center, radius, (255), thickness=-1)

    return mask.astype(bool)


def mask_center_circle(image: np.ndarray) -> np.ndarray:
    """Apply a circular mask to the center of the image. The function creates a mask that zeroes out the area except
    for the central circle, which is tightly fit to the image's shorter side among width and height.

    Args:
        image: The original image with shape (height, width [, channels]).

    Returns:
        The image with the same shape as the input, where a circular mask has
        been applied, blacking out areas outside the central circle.
    """

    # Generate center circle mask
    mask = generate_center_circle_mask(image.shape[:2])

    # Apply the mask to the image
    image = image * mask[..., np.newaxis]

    return image


def color_transfer(image: np.ndarray, color_space: RGBColorSpace) -> np.ndarray:
    """Transfer the RGB color space of an image.

    Args:
        image: The original RGB image with shape (height, width, channels).
        color_space: The RGB color space to transfer to.

    Returns:
        The image with the RGB color space transferred.
    """

    image = image.astype(np.float32)

    # Compute the original and target mean and standard deviation
    original_mean = image.mean(axis=(0, 1))
    original_std = image.std(axis=(0, 1))
    target_mean = np.array([color_space.mean.r, color_space.mean.g, color_space.mean.b])
    target_std = np.array([color_space.std.r, color_space.std.g, color_space.std.b])

    # Transfer the color space
    normalized = (image - original_mean) / (original_std + np.finfo(np.float32).eps)
    transferred = normalized * target_std + target_mean

    # Postprocess the transferred image
    transferred = np.clip(transferred, 0, 255).astype(np.uint8)
    return transferred
