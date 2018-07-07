import chainer
import chainer.links as L
from chainer.initializers import HeNormal


class CoreParameters(chainer.Chain):
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
            self.pixel_shuffle = L.Convolution2D(
                None,
                channels_u * 16,
                ksize=5,
                stride=1,
                pad=2,
                initialW=HeNormal(0.1))
            self.deconv_h = L.Deconvolution2D(
                None,
                channels_u,
                ksize=4,
                stride=4,
                pad=0,
                initialW=HeNormal(0.1))


class PriorParameters(chainer.Chain):
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


class ObservationParameters(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.mean_x = L.Convolution2D(
                None, 3, ksize=1, stride=1, pad=0, initialW=HeNormal(0.1))