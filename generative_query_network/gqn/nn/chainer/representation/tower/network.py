import chainer
import chainer.functions as cf

from .... import base
from .parameters import TowerParameters


class TowerNetwork(base.representation.Network):
    def __init__(self, params: TowerParameters):
        assert isinstance(params, TowerParameters)
        self.params = params

    def compute_r(self, x, v):
        resnet_in = cf.relu(self.params.conv1_1(x))
        residual = cf.relu(self.params.conv1_res(resnet_in))
        out = cf.relu(self.params.conv1_2(resnet_in))
        out = cf.relu(self.params.conv1_3(out)) + residual
        v = cf.reshape(v, (v.shape[0], v.shape[1], 1, 1))
        broadcast_shape = (
            out.shape[0],
            v.shape[1],
        ) + out.shape[2:]
        v = cf.broadcast_to(v, shape=broadcast_shape)
        resnet_in = cf.concat((out, v), axis=1)
        residual = cf.relu(self.params.conv2_res(resnet_in))
        out = cf.relu(self.params.conv2_1(resnet_in))
        out = cf.relu(self.params.conv2_2(out)) + residual
        out = cf.relu(self.params.conv2_3(out))
        return out
