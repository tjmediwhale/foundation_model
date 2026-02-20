import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--save-images",
        action="store_true",
        default=False,
        help="Save transformed images rather than comparing them against expected images",
    )
    parser.addoption(
        "--gpu",
        action="store_true",
        default=False,
        help="Use GPU for testing kornia library",
    )


@pytest.fixture
def random_image(request) -> np.ndarray:
    """A random RGB image of the given shape."""

    image_shape = request.param
    image = np.random.randint(low=0, high=256, size=image_shape + (3,), dtype=np.uint8)
    return image
