from . import data
from . import mathematics as math
from . import functions
from . import nn
from . import preprocessing
from chainer.backends import cuda
import numpy
import cupy


def to_device(x, device_id):
    if device_id is None:
        return x
    if device_id < 0 and isinstance(x, cupy.ndarray):
        return cuda.to_cpu(x)
    if device_id >= 0 and isinstance(x, numpy.ndarray):
        return cuda.to_gpu(x, device_id)
    return x