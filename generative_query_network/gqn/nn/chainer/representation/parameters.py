import chainer
import chainer.links as L


class Parameters(chainer.Chain):
    def __init__(self):
        super().__init__(
            conv1_1=L.Convolution2D(None, 256, ksize=2, pad=0, stride=2),
            conv1_2=L.Convolution2D(None, 128, ksize=3, pad=1, stride=1),
            conv1_3=L.Convolution2D(None, 256, ksize=2, pad=0, stride=2),
            conv2_1=L.Convolution2D(None, 128, ksize=3, pad=1, stride=1),
            conv2_2=L.Convolution2D(None, 256, ksize=3, pad=1, stride=1),
            conv2_3=L.Convolution2D(None, 256, ksize=1, pad=0, stride=1))
