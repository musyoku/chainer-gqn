import chainer
import chainer.functions as F

from ... import base
from .parameters import Parameters


class Network(base.representation.Network):
    def __init__(self, params):
        assert isinstance(params, Parameters)
        self.params = params

    def compute_r(self, x, v):
        output = x
        output = F.relu(self.params.conv1_1(output))
        output = F.relu(self.params.conv1_2(output))
        output = F.relu(self.params.conv1_3(output)) + output
        v = F.broadcast_to(v, shape=output.shape[:3] + v.shape[2])
        output = F.concat((output, v), axis=3)
        output = F.relu(self.params.conv2_1(output))
        output = F.relu(self.params.conv2_2(output)) + output
        output = F.relu(self.params.conv2_3(output))
        return output
