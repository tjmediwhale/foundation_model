import numpy as np
import torch


def convert_np_array_to_torch_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert a numpy array to a torch tensor.

    Args:
        image: The numpy array.

    Returns:
        The torch tensor.
    """

    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    image = image.unsqueeze(0)
    return image


def convert_torch_tensor_to_np_array(image: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a numpy array.

    Args:
        image: The torch tensor.

    Returns:
        The numpy array.
    """

    image = image.squeeze(0)
    image = image.permute(1, 2, 0).numpy() * 255.0
    image = image.astype(np.uint8)
    return image
