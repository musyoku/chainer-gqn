import chainer
import chainer.functions as cf
import chainer.links as nn
import cupy

from .cuda import RepresentationTowerFunction


class TowerNetwork(chainer.Chain):
    def __init__(self,
                 r_channels=256,
                 v_size=(16, 16),
                 use_cuda_kernel=False,
                 weight_initializer=None):
        super().__init__()
        self.v_size = v_size
        self.use_cuda_kernel = use_cuda_kernel
        with self.init_scope():
            self.conv1_1 = nn.Convolution2D(
                3,
                r_channels,
                ksize=2,
                pad=0,
                stride=2,
                initialW=weight_initializer)
            self.conv1_2 = nn.Convolution2D(
                r_channels,
                r_channels // 2,
                ksize=3,
                pad=1,
                stride=1,
                initialW=weight_initializer)
            self.conv1_res = nn.Convolution2D(
                r_channels,
                r_channels,
                ksize=2,
                pad=0,
                stride=2,
                initialW=weight_initializer)
            self.conv1_3 = nn.Convolution2D(
                r_channels // 2,
                r_channels,
                ksize=2,
                pad=0,
                stride=2,
                initialW=weight_initializer)
            self.conv2_1 = nn.Convolution2D(
                r_channels + 7,
                r_channels // 2,
                ksize=3,
                pad=1,
                stride=1,
                initialW=weight_initializer)
            self.conv2_2 = nn.Convolution2D(
                r_channels // 2,
                r_channels,
                ksize=3,
                pad=1,
                stride=1,
                initialW=weight_initializer)
            self.conv2_res = nn.Convolution2D(
                r_channels + 7,
                r_channels,
                ksize=3,
                pad=1,
                stride=1,
                initialW=weight_initializer)
            self.conv2_3 = nn.Convolution2D(
                r_channels,
                r_channels,
                ksize=1,
                pad=0,
                stride=1,
                initialW=weight_initializer)
            self.broadcast_v = nn.Deconvolution2D(
                7,
                7,
                ksize=self.v_size,
                pad=0,
                stride=self.v_size,
                initialW=weight_initializer)

    def __call__(self, x, v):
        v = self.broadcast_v(v)

        if self.use_cuda_kernel:
            out = RepresentationTowerFunction()(
                x, v, self.conv1_1.W, self.conv1_1.b, self.conv1_2.W,
                self.conv1_2.b, self.conv1_res.W, self.conv1_res.b,
                self.conv1_3.W, self.conv1_3.b, self.conv2_1.W, self.conv2_1.b,
                self.conv2_res.W, self.conv2_res.b, self.conv2_2.W,
                self.conv2_2.b, self.conv2_3.W, self.conv2_3.b)
        else:
            resnet_in = cf.relu(self.conv1_1(x))
            residual = cf.relu(self.conv1_res(resnet_in))
            out = cf.relu(self.conv1_2(resnet_in))
            out = cf.relu(self.conv1_3(out)) + residual
            resnet_in = cf.concat((out, v), axis=1)
            residual = cf.relu(self.conv2_res(resnet_in))
            out = cf.relu(self.conv2_1(resnet_in))
            out = cf.relu(self.conv2_2(out)) + residual
            out = self.conv2_3(out)
        return out


class PoolNetwork(TowerNetwork):
    def __init__(self,
                 r_channels=256,
                 r_size=(16, 16),
                 v_size=(16, 16),
                 use_cuda_kernel=False,
                 weight_initializer=None):
        super().__init__(
            r_channels=r_channels,
            v_size=v_size,
            use_cuda_kernel=use_cuda_kernel,
            weight_initializer=weight_initializer)
        self.r_size = r_size

    def __call__(self, x, v):
        out = super().__call__(x, v)
        out = cf.average_pooling_2d(out, self.r_size)
        return out
