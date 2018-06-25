import os
import sys
import chainer
import uuid
from chainer.serializers import load_hdf5, save_hdf5

sys.path.append(os.path.join("..", ".."))
import gqn
from hyper_parameters import HyperParameters


class Model():
    def __init__(self, hyperparams: HyperParameters, hdf5_path=None):
        assert isinstance(hyperparams, HyperParameters)

        self.generation_network, self.generation_network_params = self.build_generation_network(
            total_timestep=hyperparams.generator_total_timestep,
            channels_chz=hyperparams.channels_chz,
            channels_u=hyperparams.generator_u_channels)

        self.inference_network, self.inference_network_params = self.build_inference_network(
            channels_chz=hyperparams.channels_chz)

        self.representation_network, self.representation_network_params = self.build_representation_network(
            architecture=hyperparams.representation_architecture,
            channels_r=hyperparams.channels_r)

        if hdf5_path:
            try:
                load_hdf5(
                    os.path.join(hdf5_path, "generation.hdf5"),
                    self.generation_network_params)
                load_hdf5(
                    os.path.join(hdf5_path, "inference.hdf5"),
                    self.inference_network_params)
                load_hdf5(
                    os.path.join(hdf5_path, "representation.hdf5"),
                    self.representation_network_params)
            except:
                pass

        self.all_parameters = chainer.Chain(
            g=self.generation_network_params,
            i=self.inference_network_params,
            r=self.representation_network_params,
        )
        self.generation_parameters = chainer.Chain(
            r=self.representation_network_params,
            g=self.generation_network_params,
        )
        self.inference_parameters = chainer.Chain(
            i=self.inference_network_params, )

    def build_generation_network(self, total_timestep, channels_chz,
                                 channels_u):
        params = gqn.nn.chainer.generator.Parameters(
            channels_chz=channels_chz, channels_u=channels_u)
        network = gqn.nn.chainer.generator.Network(
            params=params, total_timestep=total_timestep)
        return network, params

    def build_inference_network(self, channels_chz):
        params = gqn.nn.chainer.inference.Parameters(
            channels_chz=channels_chz)
        network = gqn.nn.chainer.inference.Network(params=params)
        return network, params

    def build_representation_network(self, architecture, channels_r):
        if architecture == "tower":
            params = gqn.nn.chainer.representation.tower.Parameters(
                channels_r=channels_r)
            network = gqn.nn.chainer.representation.tower.Network(
                params=params)
            return network, params
        raise NotImplementedError

    def to_gpu(self):
        self.all_parameters.to_gpu()

    def cleargrads(self):
        self.all_parameters.cleargrads()

    def serialize(self, path):
        self.serialize_parameter(path, "generation.hdf5",
                                 self.generation_network_params)
        self.serialize_parameter(path, "inference.hdf5",
                                 self.inference_network_params)
        self.serialize_parameter(path, "representation.hdf5",
                                 self.representation_network_params)

    def serialize_parameter(self, path, filename, params):
        tmp_filename = str(uuid.uuid4())
        save_hdf5(
            os.path.join(path, tmp_filename), params)
        os.rename(
            os.path.join(path, tmp_filename), os.path.join(path, filename))
