import chainer
import chainer.functions as cf
import chainer.links as nn
from chainer.initializers import HeNormal


class Downsampler(chainer.Chain):
    def __init__(self, channels):
        super().__init__()
        with self.init_scope():
            self.conv = nn.Convolution2D(
                None,
                channels,
                ksize=4,
                stride=4,
                pad=0,
                initialW=HeNormal(0.1))

    def __call__(self, x):
        return self.conv(x)


class _Downsampler(chainer.Chain):
    def __init__(self, channels):
        super().__init__()
        with self.init_scope():
            self.conv_1 = nn.Convolution2D(
                None,
                channels,
                ksize=2,
                stride=2,
                pad=0,
                initialW=HeNormal(0.1))
            self.conv_2 = nn.Convolution2D(
                None,
                channels,
                ksize=3,
                pad=1,
                stride=1,
                initialW=HeNormal(0.1))
            self.conv_3 = nn.Convolution2D(
                None,
                channels,
                ksize=2,
                stride=2,
                pad=0,
                initialW=HeNormal(0.1))

    def __call__(self, x):
        out = x
        out = cf.relu(self.conv_1(out))
        out = cf.relu(self.conv_2(out))
        out = self.conv_3(out)
        return out
