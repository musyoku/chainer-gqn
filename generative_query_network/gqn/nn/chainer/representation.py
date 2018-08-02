import chainer
import chainer.functions as cf
import chainer.links as L
from chainer.initializers import HeNormal


class TowerNetwork(chainer.Chain):
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

    def compute_r(self, x, v):
        resnet_in = cf.relu(self.conv1_1(x))
        residual = cf.relu(self.conv1_res(resnet_in))
        out = cf.relu(self.conv1_2(resnet_in))
        out = cf.relu(self.conv1_3(out)) + residual
        v = cf.reshape(v, (v.shape[0], v.shape[1], 1, 1))
        broadcast_shape = (
            out.shape[0],
            v.shape[1],
        ) + out.shape[2:]
        v = cf.broadcast_to(v, shape=broadcast_shape)
        resnet_in = cf.concat((out, v), axis=1)
        residual = cf.relu(self.conv2_res(resnet_in))
        out = cf.relu(self.conv2_1(resnet_in))
        out = cf.relu(self.conv2_2(out)) + residual
        out = self.conv2_3(out)
        return out
