import chainer
import chainer.functions as cf
import cupy
import math

from ... import base
from .parameters import CoreParameters, PosteriorParameters, DownsamplerParameters
from ..functions import get_array_module


class CoreNetwork(base.inference.CoreNetwork):
    def __init__(self, params: CoreParameters):
        assert isinstance(params, CoreParameters)
        self.params = params

    def forward_onestep(self, prev_hg, prev_he, prev_ce, x, v, r):
        xp = get_array_module(v)
        broadcast_shape = (
            prev_he.shape[0],
            v.shape[1],
        ) + prev_he.shape[2:]
        v = xp.reshape(v, v.shape + (1, 1))
        v = xp.broadcast_to(v, shape=broadcast_shape)

        lstm_in = cf.concat((prev_he, prev_hg, x, v, r), axis=1)
        forget_gate = cf.sigmoid(self.params.lstm_f(lstm_in))
        input_gate = cf.sigmoid(self.params.lstm_i(lstm_in))
        next_c = forget_gate * prev_ce + input_gate * cf.tanh(
            self.params.lstm_tanh(lstm_in))
        next_h = cf.sigmoid(self.params.lstm_o(lstm_in)) * cf.tanh(next_c)
        return next_h, next_c


class PosteriorNetwork(base.inference.PosteriorNetwork):
    def __init__(self, params: PosteriorParameters):
        assert isinstance(params, PosteriorParameters)
        self.params = params

    def compute_mean_z(self, h):
        return self.params.mean_z(h)

    def compute_ln_var_z(self, h):
        return self.params.ln_var_z(h)

    def sample_z(self, h):
        mean = self.compute_mean_z(h)
        ln_var = self.compute_ln_var_z(h)
        return cf.gaussian(mean, ln_var)


class Downsampler(base.inference.Downsampler):
    def __init__(self, params: DownsamplerParameters):
        assert isinstance(params, DownsamplerParameters)
        self.params = params

    def downsample(self, x):
        x = cf.relu(self.params.conv_x_1(x))
        x = cf.relu(self.params.conv_x_2(x))
        return x
