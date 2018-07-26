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
        self.parameters = chainer.ChainList()

        self.generation_cores, self.generation_priors, self.generation_observation = self.build_generation_network(
            generation_steps=self.generation_steps,
            channels_chz=hyperparams.channels_chz,
            channels_u=hyperparams.generator_channels_u)

        self.inference_cores, self.inference_posteriors, self.inference_downsampler = self.build_inference_network(
            generation_steps=self.generation_steps,
            channels_chz=hyperparams.channels_chz,
            channels_map_x=hyperparams.inference_channels_map_x)

        self.representation_network = self.build_representation_network(
            architecture=hyperparams.representation_architecture,
            channels_r=hyperparams.channels_r)

        if hdf5_path:
            try:
                load_hdf5(
                    os.path.join(hdf5_path, "model.hdf5"), self.parameters)
            except:
                pass

    def build_generation_network(self, num_cores, channels_chz, channels_u):
        cores = []
        priors = []
        with self.parameters.init_scope():
            # LSTM core
            for _ in range(num_cores):
                core = gqn.nn.chainer.generator.Core(
                    channels_chz=channels_chz, channels_u=channels_u)
                cores.append(core)
                self.parameters.append(core)

            # z prior sampler
            for t in range(num_cores):
                prior = gqn.nn.chainer.generator.Prior(channels_z=channels_chz)
                priors.append(prior)
                self.parameters.append(prior)

            # observation sampler
            observation_distribution = gqn.nn.chainer.generator.ObservationDistribution(
            )
            self.parameters.append(observation_distribution)

        return cores, priors, observation_distribution

    def build_inference_network(self, num_cores, channels_chz, channels_map_x):
        cores = []
        posteriors = []
        with self.parameters.init_scope():
            for t in range(num_cores):
                # LSTM core
                core = gqn.nn.chainer.inference.Core(channels_chz=channels_chz)
                cores.append(core)
                self.parameters.append(core)

            # z posterior sampler
            for t in range(num_cores):
                posterior = gqn.nn.chainer.inference.Posterior(
                    channels_z=channels_chz)
                posteriors.append(posterior)
                self.parameters.append(posterior)

            # x downsampler
            params = gqn.nn.chainer.inference.DownsamplerParameters(
                channels=channels_map_x)
            downsampler = gqn.nn.chainer.inference.Downsampler(params)
            self.parameters.downsampler = params

        return cores, posteriors, downsampler

    def build_representation_network(self, architecture, channels_r):
        if architecture == "tower":
            layer = gqn.nn.chainer.representation.TowerNetwork(
                channels_r=channels_r)
            with self.parameters.init_scope():
                self.parameters.append(layer)
            return layer

        raise NotImplementedError

    def to_gpu(self):
        self.parameters.to_gpu()

    def cleargrads(self):
        self.parameters.cleargrads()

    def serialize(self, path):
        self.serialize_parameter(path, "model.hdf5", self.parameters)

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
                self.hyperparams.generator_channels_u,
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

    def get_generation_prior(self, l):
        return self.generation_priors[l]

    def get_inference_core(self, l):
        return self.inference_cores[l]

    def get_inference_posterior(self, l):
        return self.inference_posteriors[l]

    def generate_image(self, query_viewpoints, r, xp):
        batch_size = query_viewpoints.shape[0]
        h0_g, c0_g, u0, _, _ = self.generate_initial_state(batch_size, xp)
        hl_g = h0_g
        cl_g = c0_g
        ul_g = u0
        for l in range(self.generation_steps):
            core = self.get_generation_core(l)
            prior = self.get_generation_prior(l)
            zg_l = prior.sample_z(hl_g)
            next_h_g, next_c_g, next_u_g = core.forward_onestep(
                hl_g, cl_g, ul_g, zg_l, query_viewpoints, r)

            hl_g = next_h_g
            cl_g = next_c_g
            ul_g = next_u_g

        x = self.generation_observation.compute_mean_x(ul_g)
        return x.data

    def reconstruct_image(self, query_images, query_viewpoints, r, xp):
        batch_size = query_viewpoints.shape[0]
        h0_g, c0_g, u0, h0_e, c0_e = self.generate_initial_state(
            batch_size, xp)

        hl_e = h0_e
        cl_e = c0_e
        hl_g = h0_g
        cl_g = c0_g
        ul_e = u0

        xq = self.inference_downsampler.downsample(query_images)

        for l in range(self.generation_steps):
            inference_core = self.get_inference_core(l)
            inference_posterior = self.get_inference_posterior(l)
            generation_core = self.get_generation_core(l)

            he_next, ce_next = inference_core.forward_onestep(
                hl_g, hl_e, cl_e, xq, query_viewpoints, r)

            ze_l = inference_posterior.sample_z(hl_e)

            hg_next, cg_next, ue_next = generation_core.forward_onestep(
                hl_g, cl_g, ul_e, ze_l, query_viewpoints, r)

            hl_g = hg_next
            cl_g = cg_next
            ul_e = ue_next
            hl_e = he_next
            cl_e = ce_next

        x = self.generation_observation.compute_mean_x(ul_e)
        return x.data
