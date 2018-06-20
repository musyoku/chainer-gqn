import chainer
import chainer.functions as F

from .... import base
from .parameters import Parameters


class Network(base.representation.Network):
    def __init__(self, params):
        assert isinstance(params, Parameters)
        self.params = params

    def compute_r(self, x, v):
        resnet_in = F.relu(self.params.conv1_1(x))
        residual = F.relu(self.params.conv1_res(resnet_in))
        out = F.relu(self.params.conv1_2(resnet_in))
        out = F.relu(self.params.conv1_3(out)) + residual
        v = F.reshape(v, (v.shape[0], v.shape[1], 1, 1))
        broadcast_shape = (
            out.shape[0],
            v.shape[1],
        ) + out.shape[2:]
        v = F.broadcast_to(v, shape=broadcast_shape)
        resnet_in = F.concat((out, v), axis=1)
        residual = F.relu(self.params.conv2_res(resnet_in))
        out = F.relu(self.params.conv2_1(resnet_in))
        out = F.relu(self.params.conv2_2(out)) + residual
        out = F.relu(self.params.conv2_3(out))
        return out
