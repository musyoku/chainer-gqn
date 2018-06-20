import chainer
import chainer.links as L


class Parameters(chainer.Chain):
    def __init__(self, image_size=(64, 64), ndim_u=32):
        super().__init__(
            Wz=L.Convolution2D(None, 128, ksize=5, stride=1, pad=1),
            Wi=L.Convolution2D(None, 128, ksize=5, stride=1, pad=1),
            Wf=L.Convolution2D(None, 128, ksize=5, stride=1, pad=1),
            Wo=L.Convolution2D(None, 128, ksize=5, stride=1, pad=1),
            upsampler=L.Deconvolution2D(
                None, ndim_u, ksize=4, stride=4, pad=0))
