import numpy as np
import cupy as cp
from chainer.backends import cuda


def preprocess_images(images, add_noise=False):
    xp = cuda.get_array_module(images)
    if add_noise:
        images += xp.random.uniform(0, 1, size=images.shape).astype(xp.float32)
    images = images / 256 - 0.5
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
