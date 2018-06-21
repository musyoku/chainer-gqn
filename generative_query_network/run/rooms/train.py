import argparse
import sys
import os
import random
import math
import numpy as np
import cupy as xp
import chainer
import chainer.functions as cf

sys.path.append(os.path.join("..", ".."))
import gqn
from hyper_parameters import HyperParameters
from model import Model
from optimizer import Optimizer


def main():
    try:
        os.mkdir(args.snapshot_path)
    except:
        pass

    dataset = gqn.data.Dataset(args.dataset_path)
    sampler = gqn.data.Sampler(dataset)
    iterator = gqn.data.Iterator(sampler, batch_size=args.batch_size)

    hyperparams = HyperParameters()
    model = Model(hyperparams)
    model.to_gpu()

    optimizer = Optimizer(model.parameters)

    with chainer.using_config("train", True):
        for iteration in range(args.training_steps):
            for batch_index, data_indices in enumerate(iterator):
                # shape: (batch, views, height, width, channels)
                # range: [-1, 1]
                images, viewpoints = dataset[data_indices]

                image_size = images.shape[2:4]
                total_views = images.shape[1]

                # sample number of views
                num_views = random.choice(range(total_views))
                query_index = random.choice(range(total_views))

                if num_views > 0:
                    observed_images = images[:, :num_views]
                    observed_viewpoints = viewpoints[:, :num_views]

                    # (batch, views, height, width, channels) -> (batch * views, height, width, channels)
                    observed_images = observed_images.reshape((
                        args.batch_size * num_views, ) + observed_images.shape[2:])
                    observed_viewpoints = observed_viewpoints.reshape(
                        (args.batch_size * num_views, ) +
                        observed_viewpoints.shape[2:])

                    # (batch * views, height, width, channels) -> (batch * views, channels, height, width)
                    observed_images = observed_images.transpose((0, 3, 1, 2))

                    # transfer to gpu
                    observed_images = chainer.cuda.to_gpu(observed_images)
                    observed_viewpoints = chainer.cuda.to_gpu(observed_viewpoints)

                    r = model.representation_network.compute_r(
                        observed_images, observed_viewpoints)

                    # (batch * views, channels, height, width) -> (batch, views, channels, height, width)
                    r = r.reshape((args.batch_size, num_views) + r.shape[1:])

                    # sum element-wise across views
                    r = cf.sum(r, axis=1)
                else:
                    r = np.zeros(
                        (args.batch_size, hyperparams.channels_r) +
                        hyperparams.chrz_size,
                        dtype="float32")
                    r = chainer.cuda.to_gpu(r)

                query_images = images[:, query_index]
                query_viewpoints = viewpoints[:, query_index]

                # (batch * views, height, width, channels) -> (batch * views, channels, height, width)
                query_images = query_images.transpose((0, 3, 1, 2))

                # transfer to gpu
                query_images = chainer.cuda.to_gpu(query_images)
                query_viewpoints = chainer.cuda.to_gpu(query_viewpoints)

                hg_0 = xp.zeros(
                    (
                        args.batch_size,
                        hyperparams.channels_chz,
                    ) + hyperparams.chrz_size,
                    dtype="float32")
                cg_0 = xp.zeros(
                    (
                        args.batch_size,
                        hyperparams.channels_chz,
                    ) + hyperparams.chrz_size,
                    dtype="float32")
                u_0 = xp.zeros(
                    (
                        args.batch_size,
                        hyperparams.generator_u_channels,
                    ) + image_size,
                    dtype="float32")
                he_0 = xp.zeros(
                    (
                        args.batch_size,
                        hyperparams.channels_chz,
                    ) + hyperparams.chrz_size,
                    dtype="float32")
                ce_0 = xp.zeros(
                    (
                        args.batch_size,
                        hyperparams.channels_chz,
                    ) + hyperparams.chrz_size,
                    dtype="float32")

                sigma_t = 1.0

                loss_kld = 0
                he_l = he_0
                ce_l = ce_0
                hg_l = hg_0
                cg_l = cg_0
                u_l = u_0
                for l in range(hyperparams.generator_total_timestep):
                    he_next, ce_next = model.inference_network.forward_onestep(
                        hg_l, he_l, ce_l, query_images, query_viewpoints, r)
                    mu_z_q = model.inference_network.compute_mu_z(he_l)
                    ze_l = cf.gaussian(mu_z_q, xp.zeros_like(mu_z_q))
                    hg_next, cg_next, u_next = model.generation_network.forward_onestep(
                        hg_l, cg_l, u_l, ze_l, query_viewpoints, r)
                    mu_z_p = model.generation_network.compute_mu_z(hg_l)

                    kld = gqn.nn.chainer.functions.gaussian_kl_divergence(
                        mu_z_q, mu_z_p)

                    loss_kld += cf.mean(kld)

                    hg_l = hg_next
                    cg_l = cg_next
                    u_l = u_next
                    he_l = he_next
                    ce_l = ce_next

                mu_x = model.generation_network.compute_mu_x(u_l)
                negative_log_likelihood = gqn.nn.chainer.functions.gaussian_negative_log_likelihood(
                    query_images, mu_x, xp.full_like(mu_x, math.log(sigma_t)))
                loss_nll = cf.mean(negative_log_likelihood)
                loss = loss_nll + loss_kld
                model.cleargrads()
                loss.backward()
                optimizer.step()

                print("Iteration {}: {} / {} - loss: {}".format(
                    iteration + 1, batch_index + 1, len(iterator),
                    float(loss.data)))

            model.serialize(args.snapshot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="rooms_dataset")
    parser.add_argument("--snapshot-path", type=str, default="snapshot")
    parser.add_argument("--batch-size", "-b", type=int, default=36)
    parser.add_argument(
        "--training-steps", "-smax", type=int, default=2 * 10**6)
    args = parser.parse_args()
    main()
