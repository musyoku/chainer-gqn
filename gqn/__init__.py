import cupy
import numpy
from chainer import cuda

from . import data, nn, preprocessing, json


def to_device(x, device_id):
    if device_id is None:
        return x
    if device_id < 0 and isinstance(x, cupy.ndarray):
        return cuda.to_cpu(x)
    if device_id >= 0 and isinstance(x, numpy.ndarray):
        return cuda.to_gpu(x, device_id)
    return x
