import chainer
import chainer.functions as cf
import chainer.links as nn
from chainer.initializers import HeNormal


class SingleConvDownsampler(chainer.Chain):
    def __init__(self, channels):
        super().__init__()
        with self.init_scope():
            self.conv = nn.Convolution2D(
                None,
                channels,
                ksize=6,
                stride=4,
                pad=1,
                initialW=HeNormal(0.1))

    def __call__(self, x):
        return self.conv(x)
