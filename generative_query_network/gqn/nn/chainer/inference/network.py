import chainer
import chainer.functions as cf
import cupy
import math

from ... import base
from .parameters import Parameters
from ..functions import get_array_module


class Network(base.inference.Network):
    def __init__(self, params):
        assert isinstance(params, Parameters)
        self.params = params

    def forward_onestep(self, prev_h_g, prev_h_e, prev_c_e, x, v, r):
        xp = get_array_module(v)
        broadcast_shape = (
            prev_h_e.shape[0],
            v.shape[1],
        ) + prev_h_e.shape[2:]
        v = xp.reshape(v, v.shape + (1, 1))
        v = xp.broadcast_to(v, shape=broadcast_shape)

        # x = cf.relu(self.params.conv_x(x))
        x = cf.average_pooling_2d(x, ksize=4)

        lstm_in = cf.concat((prev_h_e, prev_h_g, x, v, r), axis=1)
        forget_gate = cf.sigmoid(self.params.lstm_f(lstm_in))
        input_gate = cf.sigmoid(self.params.lstm_i(lstm_in))
        next_c = forget_gate * prev_c_e + input_gate * cf.tanh(
            self.params.lstm_tanh(lstm_in))
        next_h = cf.sigmoid(self.params.lstm_o(lstm_in)) * cf.tanh(next_c)
        return next_h, next_c

    def compute_mean_z(self, h):
        return self.params.mean_z(h)

    def compute_ln_var_z(self, h):
        return self.params.ln_var_z(h)

    def sample_z(self, h):
        xp = get_array_module(h)
        mean = self.compute_mean_z(h)
        ln_var = self.compute_ln_var_z(h)
        return cf.gaussian(mean, ln_var)
