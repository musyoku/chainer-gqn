import chainer
import chainer.functions as cf
import cupy
import math

from ... import base
from .parameters import CoreParameters, PriorParameters, ObservationParameters
from ..functions import get_array_module


class CoreNetwork(base.generator.CoreNetwork):
    def __init__(self, params: CoreParameters):
        assert isinstance(params, CoreParameters)
        self.params = params

    def forward_onestep(self, prev_hg, prev_cg, prev_u, prev_z, v, r):
        xp = get_array_module(v)
        broadcast_shape = (
            prev_hg.shape[0],
            v.shape[1],
        ) + prev_hg.shape[2:]
        v = xp.reshape(v, v.shape + (1, 1))
        v = xp.broadcast_to(v, shape=broadcast_shape)

        lstm_in = cf.concat((prev_hg, v, r, prev_z), axis=1)
        forget_gate = cf.sigmoid(self.params.lstm_f(lstm_in))
        input_gate = cf.sigmoid(self.params.lstm_i(lstm_in))
        next_c = forget_gate * prev_cg + input_gate * cf.tanh(
            self.params.lstm_tanh(lstm_in))
        next_h = cf.sigmoid(self.params.lstm_o(lstm_in)) * cf.tanh(next_c)

        next_u = self.params.deconv_h(next_h) + prev_u
        # next_u = self.upsample_h(next_h) + prev_u

        return next_h, next_c, next_u

    # pixel shuffler
    def upsample_h(self, h):
        r = 4
        out = self.params.pixel_shuffle(h)  # 畳み込み
        batchsize = out.shape[0]
        in_channels = out.shape[1]
        out_channels = in_channels // (r**2)
        in_height = out.shape[2]
        in_width = out.shape[3]
        out_height = in_height * r
        out_width = in_width * r
        out = cf.reshape(out,
                         (batchsize, r, r, out_channels, in_height, in_width))
        out = cf.transpose(out, (0, 3, 4, 1, 5, 2))
        out = cf.reshape(out, (batchsize, out_channels, out_height, out_width))
        return out


class PriorNetwork(base.generator.PriorNetwork):
    def __init__(self, params: PriorParameters):
        assert isinstance(params, PriorParameters)
        self.params = params

    def compute_mean_z(self, h):
        return self.params.mean_z(h)

    def compute_ln_var_z(self, h):
        return self.params.ln_var_z(h)

    def sample_z(self, h):
        mean = self.compute_mean_z(h)
        ln_var = self.compute_ln_var_z(h)
        return cf.gaussian(mean, ln_var)


class ObservationNetwork(base.generator.ObservationNetwork):
    def __init__(self, params: ObservationParameters):
        assert isinstance(params, ObservationParameters)
        self.params = params

    def compute_mean_x(self, u):
        return self.params.mean_x(u)

    def sample_x(self, u, ln_var):
        mean = self.compute_mean_x(u)
        return cf.gaussian(mean, ln_var)