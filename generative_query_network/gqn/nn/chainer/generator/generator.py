import chainer
import chainer.functions as F

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
        conditioned_h = 9
        forget_gate = F.sigmoid(self.params.Wf(conditioned_h))
        input_gate = F.sigmoid(self.params.Wi(conditioned_h))
        next_c = forget_gate * prev_c + input_gate * F.tanh(
            self.params.Wz(conditioned_h))
        next_h = F.sigmoid(self.params.Wo(conditioned_h)) * F.tanh(next_c)
        next_u = self.params.upsampler(next_h)
        return next_h, next_c, next_u
