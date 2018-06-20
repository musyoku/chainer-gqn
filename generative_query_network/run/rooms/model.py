import os
import sys

sys.path.append(os.path.join("..", ".."))
import gqn
from hyper_parameters import HyperParameters


class Model():
    def __init__(self, hyperparams: HyperParameters):
        assert isinstance(hyperparams, HyperParameters)

        self.generation_network, self.generation_network_params = self.build_generation_network(
            total_timestep=hyperparams.generator_total_timestep,
            channels_chrz=hyperparams.chrz_channels,
            channels_u=hyperparams.generator_u_channels,
            sigma_t=hyperparams.generator_sigma_t)

        self.representation_network, self.representation_network_params = self.build_representation_network(
            architecture=hyperparams.representation_architecture)

    def build_generation_network(self, total_timestep, channels_chrz,
                                 channels_u, sigma_t):
        params = gqn.nn.chainer.generator.Parameters(
            channels_chrz=channels_chrz,
            channels_u=channels_u,
            sigma_t=sigma_t)
        network = gqn.nn.chainer.generator.Network(
            params=params, total_timestep=total_timestep)
        return network, params

    def build_representation_network(self, architecture):
        if architecture == "tower":
            params = gqn.nn.chainer.representation.tower.Parameters()
            network = gqn.nn.chainer.representation.tower.Network(
                params=params)
            return network, params
        raise NotImplementedError