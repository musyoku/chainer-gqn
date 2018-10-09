import chainer
import chainer.functions as cf
import chainer.links as nn
from chainer.initializers import HeNormal


class SingleConvDownsampler(chainer.Chain):
    def __init__(self, channels):
        super().__init__()
        with self.init_scope():
            self.conv_1 = nn.Convolution2D(
                None,
                channels // 4,
                ksize=2,
                stride=2,
                pad=0,
                initialW=HeNormal(0.1))
            self.conv_2 = nn.Convolution2D(
                None,
                channels // 2,
                ksize=2,
                stride=2,
                pad=0,
                initialW=HeNormal(0.1))
            self.conv_3 = nn.Convolution2D(
                None,
                channels,
                ksize=3,
                stride=1,
                pad=1,
                initialW=HeNormal(0.1))
            self.conv_4 = nn.Convolution2D(
                None,
                channels,
                ksize=3,
                stride=1,
                pad=1,
                initialW=HeNormal(0.1))

    def __call__(self, x):
        out = x
        out = cf.relu(self.conv_1(out))
        out = cf.relu(self.conv_2(out))
        out = cf.relu(self.conv_3(out))
        out = self.conv_4(out)
        return out
