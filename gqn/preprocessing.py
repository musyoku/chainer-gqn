import numpy as np
import cupy as cp
from chainer.backends import cuda


def preprocess_images(images, add_noise=True):
    xp = cuda.get_array_module(images)
    images = images / 255 - 0.5
    if add_noise:
        images += xp.random.uniform(
            -1.0 / 255 / 2, 1.0 / 255 / 2,
            size=images.shape).astype(np.float32)
    return images


def to_cpu(array):
    if isinstance(array, cp.ndarray):
        return cuda.to_cpu(array)
    return array


def make_uint8(image):
    assert image.ndim == 3
    if (image.shape[0] == 3):
        image = image.transpose(1, 2, 0)
    image = to_cpu(image)
    image = 255 * (image + 0.5)
    return np.uint8(np.clip(image, 0, 255))
