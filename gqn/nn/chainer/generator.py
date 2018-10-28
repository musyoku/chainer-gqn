import math

import chainer
import chainer.functions as cf
import chainer.links as nn
import cupy
from chainer.backends import cuda
from chainer.initializers import HeNormal


class Core(chainer.Chain):
    def __init__(self, h_channels):
        super().__init__()
        with self.init_scope():
            self.lstm = nn.Convolution2D(
                None,
                h_channels * 4,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1))

    def __call__(self, prev_hg, prev_cg, prev_z, v, r):
        lstm_input = cf.concat((prev_hg, v, r, prev_z), axis=1)
        gate_input = self.lstm(lstm_input)
        forget_gate_input, input_gate_input, tanh_input, output_gate_input = cf.split_axis(
            gate_input, 4, axis=1)

        forget_gate = cf.sigmoid(forget_gate_input)
        input_gate = cf.sigmoid(input_gate_input)
        next_c = forget_gate * prev_cg + input_gate * cf.tanh(tanh_input)
        output_gate = cf.sigmoid(output_gate_input)
        next_h = output_gate * cf.tanh(next_c)

        return next_h, next_c


class Prior(chainer.Chain):
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