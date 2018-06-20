import chainer
import chainer.links as L


class Parameters(chainer.Chain):
    def __init__(self, channels_chrz, channels_u, sigma_t):
        super().__init__(
            lstm_tanh=L.Convolution2D(
                None, channels_chrz, ksize=5, stride=1, pad=2),
            lstm_i=L.Convolution2D(
                None, channels_chrz, ksize=5, stride=1, pad=2),
            lstm_f=L.Convolution2D(
                None, channels_chrz, ksize=5, stride=1, pad=2),
            lstm_o=L.Convolution2D(
                None, channels_chrz, ksize=5, stride=1, pad=2),
            mean_z=L.Convolution2D(
                None, channels_chrz, ksize=5, stride=1, pad=2),
            mean_x=L.Convolution2D(None, 3, ksize=1, stride=1, pad=1),
            deconv_h=L.Deconvolution2D(
                None, channels_u, ksize=4, stride=4, pad=0))
        self.sigma_t = sigma_t
