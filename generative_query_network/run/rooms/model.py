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
        self.generation_steps = hyperparams.generator_generation_steps
        self.hyperparams = hyperparams

        self.generation_cores, self.generation_prior, self.generation_observation, self.generation_params = self.build_generation_network(
            generation_steps=self.generation_steps,
            channels_chz=hyperparams.channels_chz,
            channels_u=hyperparams.generator_u_channels)

        self.inference_cores, self.inference_posterior, self.inference_downsampler, self.inference_params = self.build_inference_network(
            generation_steps=self.generation_steps,
            channels_chz=hyperparams.channels_chz)

        self.representation_network, self.representation_params = self.build_representation_network(
            architecture=hyperparams.representation_architecture,
            channels_r=hyperparams.channels_r)

        if hdf5_path:
            try:
                load_hdf5(
                    os.path.join(hdf5_path, "generation.hdf5"),
                    self.generation_params)
                load_hdf5(
                    os.path.join(hdf5_path, "inference.hdf5"),
                    self.inference_params)
                load_hdf5(
                    os.path.join(hdf5_path, "representation.hdf5"),
                    self.representation_params)
            except:
                pass

        self.parameters = chainer.Chain()
        with self.parameters.init_scope():
            self.parameters.generation = self.generation_params
            self.parameters.inference = self.inference_params
            self.parameters.representation = self.representation_params

    def build_generation_network(self, generation_steps, channels_chz,
                                 channels_u):
        generation_parameters = chainer.Chain()
        cores = []
        with generation_parameters.init_scope():
            # LSTM core
            for t in range(generation_steps):
                params = gqn.nn.chainer.generator.CoreParameters(
                    channels_chz=channels_chz, channels_u=channels_u)
                core = gqn.nn.chainer.generator.CoreNetwork(params=params)
                cores.append(core)
                setattr(generation_parameters, "core_%d" % t, params)

            # z prior sampler
            params = gqn.nn.chainer.generator.PriorParameters(
                channels_z=channels_chz)
            prior = gqn.nn.chainer.generator.PriorNetwork(params)
            generation_parameters.prior = params

            # observation sampler
            params = gqn.nn.chainer.generator.ObservationParameters()
            observation = gqn.nn.chainer.generator.ObservationNetwork(params)
            generation_parameters.observation = params

        return cores, prior, observation, generation_parameters

    def build_inference_network(self, generation_steps, channels_chz):
        inference_parameters = chainer.Chain()
        cores = []
        with inference_parameters.init_scope():
            # LSTM core
            for t in range(generation_steps):
                params = gqn.nn.chainer.inference.CoreParameters(
                    channels_chz=channels_chz)
                core = gqn.nn.chainer.inference.CoreNetwork(params=params)
                cores.append(core)
                setattr(inference_parameters, "core_%d" % t, params)

            # z posterior sampler
            params = gqn.nn.chainer.inference.PosteriorParameters(
                channels_z=channels_chz)
            posterior = gqn.nn.chainer.inference.PosteriorNetwork(params)
            inference_parameters.posterior = params

            # x downsampler
            params = gqn.nn.chainer.inference.DownsamplerParameters(
                channels_chz=channels_chz)
            downsampler = gqn.nn.chainer.inference.Downsampler(params)
            inference_parameters.downsampler = params

        return cores, posterior, downsampler, inference_parameters

    def build_representation_network(self, architecture, channels_r):
        if architecture == "tower":
            params = gqn.nn.chainer.representation.TowerParameters(
                channels_r=channels_r)
            network = gqn.nn.chainer.representation.TowerNetwork(
                params=params)
            return network, params

        raise NotImplementedError

    def to_gpu(self):
        self.parameters.to_gpu()

    def cleargrads(self):
        self.parameters.cleargrads()

    def serialize(self, path):
        self.serialize_parameter(path, "generation.hdf5",
                                 self.generation_params)
        self.serialize_parameter(path, "inference.hdf5", self.inference_params)
        self.serialize_parameter(path, "representation.hdf5",
                                 self.representation_params)

    def serialize_parameter(self, path, filename, params):
        tmp_filename = str(uuid.uuid4())
        save_hdf5(os.path.join(path, tmp_filename), params)
        os.rename(
            os.path.join(path, tmp_filename), os.path.join(path, filename))

    def generate_initial_state(self, batch_size, xp):
        h0_g = xp.zeros(
            (
                batch_size,
                self.hyperparams.channels_chz,
            ) + self.hyperparams.chrz_size,
            dtype="float32")
        c0_g = xp.zeros(
            (
                batch_size,
                self.hyperparams.channels_chz,
            ) + self.hyperparams.chrz_size,
            dtype="float32")
        u0 = xp.zeros(
            (
                batch_size,
                self.hyperparams.generator_u_channels,
            ) + self.hyperparams.image_size,
            dtype="float32")
        h0_e = xp.zeros(
            (
                batch_size,
                self.hyperparams.channels_chz,
            ) + self.hyperparams.chrz_size,
            dtype="float32")
        c0_e = xp.zeros(
            (
                batch_size,
                self.hyperparams.channels_chz,
            ) + self.hyperparams.chrz_size,
            dtype="float32")
        return h0_g, c0_g, u0, h0_e, c0_e

    def get_generation_core(self, l):
        return self.generation_cores[l]

    def get_inference_core(self, l):
        return self.inference_cores[l]

    def generate_image(self, query_viewpoints, r, xp):
        batch_size = query_viewpoints.shape[0]
        h0_g, c0_g, u0, _, _ = self.generate_initial_state(batch_size, xp)
        hl_g = h0_g
        cl_g = c0_g
        ul_g = u0
        for l in range(self.generation_steps):
            core = self.get_generation_core(l)
            zg_l = self.generation_prior.sample_z(hl_g)
            next_h_g, next_c_g, next_u_g = core.forward_onestep(
                hl_g, cl_g, ul_g, zg_l, query_viewpoints, r)

            hl_g = next_h_g
            cl_g = next_c_g
            ul_g = next_u_g

        x = self.generation_observation.compute_mean_x(ul_g)
        return x.data
