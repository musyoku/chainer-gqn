import chainer
import chainer.links as L


class Parameters(chainer.Chain):
    def __init__(self, channels_chrz):
        super().__init__(
            lstm_tanh=L.Convolution2D(
                None, channels_chrz, ksize=5, stride=1, pad=2),
            lstm_i=L.Convolution2D(
                None, channels_chrz, ksize=5, stride=1, pad=2),
            lstm_f=L.Convolution2D(
                None, channels_chrz, ksize=5, stride=1, pad=2),
            lstm_o=L.Convolution2D(
                None, channels_chrz, ksize=5, stride=1, pad=2),
            conv_x_1=L.Convolution2D(None, 32, ksize=2, stride=2, pad=0),
            conv_x_2=L.Convolution2D(None, 64, ksize=2, stride=2, pad=0),
            mean_z=L.Convolution2D(
                None, channels_chrz, ksize=5, stride=1, pad=2))
