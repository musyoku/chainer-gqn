import chainer
import chainer.links as L
from chainer.initializers import HeNormal


class TowerParameters(chainer.Chain):
    def __init__(self, channels_r):
        super().__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(
                None,
                channels_r,
                ksize=2,
                pad=0,
                stride=2,
                initialW=HeNormal(0.1))
            self.conv1_2 = L.Convolution2D(
                None,
                channels_r // 2,
                ksize=3,
                pad=1,
                stride=1,
                initialW=HeNormal(0.1))
            self.conv1_res = L.Convolution2D(
                None,
                channels_r,
                ksize=2,
                pad=0,
                stride=2,
                initialW=HeNormal(0.1))
            self.conv1_3 = L.Convolution2D(
                None,
                channels_r,
                ksize=2,
                pad=0,
                stride=2,
                initialW=HeNormal(0.1))
            self.conv2_1 = L.Convolution2D(
                None,
                channels_r // 2,
                ksize=3,
                pad=1,
                stride=1,
                initialW=HeNormal(0.1))
            self.conv2_2 = L.Convolution2D(
                None,
                channels_r,
                ksize=3,
                pad=1,
                stride=1,
                initialW=HeNormal(0.1))
            self.conv2_res = L.Convolution2D(
                None,
                channels_r,
                ksize=3,
                pad=1,
                stride=1,
                initialW=HeNormal(0.1))
            self.conv2_3 = L.Convolution2D(
                None,
                channels_r,
                ksize=1,
                pad=0,
                stride=1,
                initialW=HeNormal(0.1))
