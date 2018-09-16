import math

import chainer
import chainer.functions as cf
import chainer.links as L
import cupy
from chainer.backends import cuda
from chainer.initializers import HeNormal


class Core(chainer.Chain):
    def __init__(self, channels_chz, channels_u):
        super().__init__()
        with self.init_scope():
            self.lstm_tanh = L.Convolution2D(
                None,
                channels_chz,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1))
            self.lstm_i = L.Convolution2D(
                None,
                channels_chz,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1))
            self.lstm_f = L.Convolution2D(
                None,
                channels_chz,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1))
            self.lstm_o = L.Convolution2D(
                None,
                channels_chz,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1))
            # self.deconv_h = L.Deconvolution2D(
            #     None,
            #     channels_u,
            #     ksize=4,
            #     stride=4,
            #     pad=0,
            #     initialW=HeNormal(0.1))
            self.conv_pixel_shuffle = L.Convolution2D(
                None,
                channels_u * 4 * 4,
                ksize=1,
                stride=1,
                pad=0,
                initialW=HeNormal(0.1))

    def forward_onestep(self, prev_hg, prev_cg, prev_u, prev_z, v, r):
        xp = cuda.get_array_module(v)
        broadcast_shape = (
            prev_hg.shape[0],
            v.shape[1],
        ) + prev_hg.shape[2:]
        v = xp.reshape(v, v.shape + (1, 1))
        v = xp.broadcast_to(v, shape=broadcast_shape)

        lstm_in = cf.concat((prev_hg, v, r, prev_z), axis=1)
        forget_gate = cf.sigmoid(self.lstm_f(lstm_in))
        input_gate = cf.sigmoid(self.lstm_i(lstm_in))
        next_c = forget_gate * prev_cg + input_gate * cf.tanh(
            self.lstm_tanh(lstm_in))
        next_h = cf.sigmoid(self.lstm_o(lstm_in)) * cf.tanh(next_c)
        # next_u = self.deconv_h(next_h) + prev_u
        next_u = cf.depth2space(self.conv_pixel_shuffle(next_h), r=4) + prev_u

        return next_h, next_c, next_u


class Prior(chainer.Chain):
    def __init__(self, channels_z):
        super().__init__()
        with self.init_scope():
            self.mean_z = L.Convolution2D(
                None,
                channels_z,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1))
            self.ln_var_z = L.Convolution2D(
                None,
                channels_z,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1))

    def compute_mean_z(self, h):
        return self.mean_z(h)

    def compute_ln_var_z(self, h):
        return self.ln_var_z(h)

    def sample_z(self, h):
        mean = self.compute_mean_z(h)
        ln_var = self.compute_ln_var_z(h)
        return cf.gaussian(mean, ln_var)


class ObservationDistribution(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.mean_x = L.Convolution2D(
                None, 3, ksize=1, stride=1, pad=0, initialW=HeNormal(0.1))

    def compute_mean_x(self, u):
        return self.mean_x(u)

    def sample_x(self, u, ln_var):
        mean = self.compute_mean_x(u)
        return cf.gaussian(mean, ln_var)
