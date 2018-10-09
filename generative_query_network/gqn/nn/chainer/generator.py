import math

import chainer
import chainer.functions as cf
import chainer.links as nn
import cupy
from chainer.backends import cuda
from chainer.initializers import HeNormal


class Core(chainer.Chain):
    def __init__(self, chz_channels):
        super().__init__()
        with self.init_scope():
            self.lstm_tanh = nn.Convolution2D(
                None,
                chz_channels,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1))
            self.lstm_i = nn.Convolution2D(
                None,
                chz_channels,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1))
            self.lstm_f = nn.Convolution2D(
                None,
                chz_channels,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1))
            self.lstm_o = nn.Convolution2D(
                None,
                chz_channels,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1))

    def __call__(self, prev_hg, prev_cg, prev_z, v, r):
        xp = cuda.get_array_module(v)
        broadcast_shape = (
            prev_hg.shape[0],
            v.shape[1],
        ) + prev_hg.shape[2:]
        v = xp.reshape(v, v.shape + (1, 1))
        v = xp.broadcast_to(v, shape=broadcast_shape)

        lstm_in = cf.concat((prev_hg, v, r, prev_z), axis=1)
        lstm_in_peephole = cf.concat((lstm_in, prev_cg), axis=1)
        forget_gate = cf.sigmoid(self.lstm_f(lstm_in_peephole))
        input_gate = cf.sigmoid(self.lstm_i(lstm_in_peephole))
        next_c = forget_gate * prev_cg + input_gate * cf.tanh(
            self.lstm_tanh(lstm_in))
        lstm_in_peephole = cf.concat((lstm_in, next_c), axis=1)
        output_gate = cf.sigmoid(self.lstm_o(lstm_in_peephole))
        next_h = output_gate * cf.tanh(next_c)

        return next_h, next_c


class Prior(chainer.Chain):
    def __init__(self, channels_z):
        super().__init__()
        self.channels_z = channels_z
        with self.init_scope():
            self.conv = nn.Convolution2D(
                None,
                channels_z * 2,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1))

    def compute_parameter(self, h):
        param = self.conv(h)
        mean = param[:, :self.channels_z]
        ln_var = param[:, self.channels_z:]
        return mean, ln_var

    def sample_z(self, h):
        mean, ln_var = self.compute_parameter(h)
        return cf.gaussian(mean, ln_var)