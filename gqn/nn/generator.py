import math
import time

import chainer
import chainer.functions as cf
import chainer.links as nn
import cupy as cp
import numpy as np
from chainer.backends import cuda

from .cuda import CoreFunction


class Core(chainer.Chain):
    def __init__(self,
                 h_channels=128,
                 h_size=(16, 16),
                 r_channels=256,
                 r_size=(16, 16),
                 u_channels=128,
                 use_cuda_kernel=False,
                 weight_initializer=None):
        super().__init__()
        self.use_cuda_kernel = use_cuda_kernel
        with self.init_scope():
            self.downsample_u = nn.Convolution2D(
                None,
                u_channels,
                ksize=4,
                stride=4,
                pad=0,
                initialW=weight_initializer)
            self.broadcast_v = nn.Deconvolution2D(
                7,
                7,
                ksize=h_size,
                pad=0,
                stride=h_size,
                initialW=weight_initializer)
            if r_size[0] == 1:
                self.broadcast_r = nn.Deconvolution2D(
                    r_channels,
                    r_channels,
                    ksize=h_size,
                    pad=0,
                    stride=h_size,
                    initialW=weight_initializer)
            self.lstm = nn.Convolution2D(
                None,
                h_channels * 4,
                ksize=5,
                stride=1,
                pad=2,
                initialW=weight_initializer)

    def __call__(self, prev_hg, prev_cg, prev_z, v, r, u):
        u = self.downsample_u(u)
        v = self.broadcast_v(v)
        if r.shape[2] == 1:
            r = self.broadcast_r(r)

        lstm_input = cf.concat((prev_hg, v, r, prev_z, u), axis=1)
        gate_inputs = self.lstm(lstm_input)

        if self.use_cuda_kernel:
            next_h, next_c = CoreFunction()(gate_inputs, prev_cg)
        else:
            forget_gate_input, input_gate_input, tanh_input, output_gate_input = cf.split_axis(
                gate_inputs, 4, axis=1)

            forget_gate = cf.sigmoid(forget_gate_input)
            input_gate = cf.sigmoid(input_gate_input)
            next_c = forget_gate * prev_cg + input_gate * cf.tanh(tanh_input)
            output_gate = cf.sigmoid(output_gate_input)
            next_h = output_gate * cf.tanh(next_c)

        return next_h, next_c


class Prior(chainer.Chain):
    def __init__(self, z_channels, weight_initializer=None):
        super().__init__()
        with self.init_scope():
            self.conv = nn.Convolution2D(
                None,
                z_channels * 2,
                ksize=5,
                stride=1,
                pad=2,
                initialW=weight_initializer)

    def compute_parameter(self, h):
        param = self.conv(h)
        mean, ln_var = cf.split_axis(param, 2, axis=1)
        return mean, ln_var

    def sample_z(self, h):
        mean, ln_var = self.compute_parameter(h)
        return cf.gaussian(mean, ln_var)
