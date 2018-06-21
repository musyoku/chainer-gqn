import chainer
import chainer.functions as cf
import cupy
import math

from ... import base
from .parameters import Parameters


class Network(base.generator.Network):
    def __init__(self, params, total_timestep=12):
        assert isinstance(params, Parameters)
        self.params = params
        self.total_timestep = total_timestep

    def generate_x(self, v, r):
        pass

    def forward_onestep(self, prev_h, prev_c, prev_u, prev_z, v, r):
        broadcast_shape = (
            prev_h.shape[0],
            v.shape[1],
        ) + prev_h.shape[2:]
        v = cf.reshape(v, v.shape + (1, 1))
        v = cf.broadcast_to(v, shape=broadcast_shape)
        lstm_in = cf.concat((prev_h, v, r, prev_z), axis=1)
        forget_gate = cf.sigmoid(self.params.lstm_f(lstm_in))
        input_gate = cf.sigmoid(self.params.lstm_i(lstm_in))
        next_c = forget_gate * prev_c + input_gate * cf.tanh(
            self.params.lstm_tanh(lstm_in))
        next_h = cf.sigmoid(self.params.lstm_o(lstm_in)) * cf.tanh(next_c)
        next_u = self.params.deconv_h(next_h) + prev_u
        return next_h, next_c, next_u

    def compute_mu_z(self, h):
        xp = cupy.get_array_module(h.data)
        mean = self.params.mean_z(h)
        return mean

    def sample_z(self, h):
        xp = cupy.get_array_module(h)
        mean = self.compute_mu_z(h)
        return cf.gaussian(mean, xp.zeros_like(mean))

    def compute_mu_x(self, u):
        xp = cupy.get_array_module(u.data)
        mean = self.params.mean_x(u)
        return mean

    def sample_x(self, u, sigma_t):
        xp = cupy.get_array_module(u.data)
        mean = self.compute_mu_x(u)
        return cf.gaussian(mean,
                           xp.full_like(mean, math.log(sigma_t)))
