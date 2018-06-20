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
            image_size=hyperparams.image_size,
            ndim_u=hyperparams.generator_ndim_u)

        self.representation_network, self.representation_network_params = self.build_representation_network(
            architecture=hyperparams.representation_architecture)

    def build_generation_network(self, total_timestep, image_size, ndim_u):
        params = gqn.nn.chainer.generator.Parameters(
            image_size=image_size, ndim_u=ndim_u)
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