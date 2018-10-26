import math

import chainer
import chainer.functions as cf
import chainer.links as nn
import cupy
from chainer.backends import cuda
from chainer.initializers import HeNormal


class Core(chainer.Chain):
    def __init__(self, h_channels, peephole_enabled=False):
        super().__init__()
        self.peephole_enabled = peephole_enabled
        with self.init_scope():
            self.lstm_tanh = nn.Convolution2D(
                None,
                h_channels,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1))
            self.lstm_i = nn.Convolution2D(
                None,
                h_channels,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1))
            self.lstm_f = nn.Convolution2D(
                None,
                h_channels,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1))
            self.lstm_o = nn.Convolution2D(
                None,
                h_channels,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1))

    def __call__(self, prev_hg, prev_he, prev_ce, downsampled_x, v, r):
        if self.peephole_enabled:
            lstm_in = cf.concat(
                (prev_he, prev_hg, downsampled_x, v, r), axis=1)
            lstm_in_peephole = cf.concat((lstm_in, prev_ce), axis=1)
            forget_gate = cf.sigmoid(self.lstm_f(lstm_in_peephole))
            input_gate = cf.sigmoid(self.lstm_i(lstm_in_peephole))
            next_c = forget_gate * prev_ce + input_gate * cf.tanh(
                self.lstm_tanh(lstm_in))
            lstm_in_peephole = cf.concat((lstm_in, next_c), axis=1)
            output_gate = cf.sigmoid(self.lstm_o(lstm_in_peephole))
            next_h = output_gate * cf.tanh(next_c)
        else:
            lstm_in = cf.concat(
                (prev_he, prev_hg, downsampled_x, v, r), axis=1)
            forget_gate = cf.sigmoid(self.lstm_f(lstm_in))
            input_gate = cf.sigmoid(self.lstm_i(lstm_in))
            next_c = forget_gate * prev_ce + input_gate * cf.tanh(
                self.lstm_tanh(lstm_in))
            output_gate = cf.sigmoid(self.lstm_o(lstm_in))
            next_h = output_gate * cf.tanh(next_c)

        return next_h, next_c


class Posterior(chainer.Chain):
    def __init__(self, z_channels):
        super().__init__()
        self.z_channels = z_channels
        with self.init_scope():
            self.conv = nn.Convolution2D(
                None,
                z_channels * 2,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1))

    def compute_parameter(self, h):
        param = self.conv(h)
        mean = param[:, :self.z_channels]
        ln_var = param[:, self.z_channels:]
        return mean, ln_var

    def sample_z(self, h):
        mean, ln_var = self.compute_parameter(h)
        return cf.gaussian(mean, ln_var)