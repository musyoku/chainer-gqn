import argparse
import sys
import os
import random
import math
import time
import numpy as np
import cupy as xp
import chainer
import chainer.functions as cf

sys.path.append(os.path.join("..", "..", ".."))
import gqn

sys.path.append(os.path.join(".."))
from hyper_parameters import HyperParameters
from model import Model


def main():
    dataset = gqn.data.Dataset(args.dataset_path)
    sampler = gqn.data.Sampler(dataset)
    iterator = gqn.data.Iterator(sampler, batch_size=args.batch_size)

    hyperparams = HyperParameters()
    model = Model(hyperparams, hdf5_path=args.snapshot_path)
    model.to_gpu()

    figure = gqn.imgplot.Figure()
    axes = []
    sqrt_batch_size = int(math.sqrt(args.batch_size))
    axis_size = 1.0 / sqrt_batch_size
    for y in range(sqrt_batch_size):
        for x in range(sqrt_batch_size * 2):
            axis = gqn.imgplot.ImageData(hyperparams.image_size[0],
                                         hyperparams.image_size[1], 3)
            axes.append(axis)
            figure.add(axis, axis_size / 2 * x, axis_size * y, axis_size / 2,
                       axis_size)
    window = gqn.imgplot.Window(figure, (1600, 800))
    window.show()

    sigma_t = hyperparams.pixel_sigma_f
    pixel_ln_var = xp.full(
        (args.batch_size, 3) + hyperparams.image_size,
        math.log(sigma_t**2),
        dtype="float32")

    with chainer.using_config("train", False), chainer.using_config(
            "enable_backprop", False):
        for data_indices in iterator:
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

            kld = 0
            he_l = he_0
            ce_l = ce_0
            hg_l = hg_0
            cg_l = cg_0
            u_l = u_0
            for l in range(hyperparams.generator_total_timestep):
                he_next, ce_next = model.inference_network.forward_onestep(
                    hg_l, he_l, ce_l, query_images, query_viewpoints, r)

                mu_z_q = model.inference_network.compute_mu_z(he_l)
                mu_z_p = model.generation_network.compute_mu_z(hg_l)
                kld += cf.mean(gqn.nn.chainer.functions.gaussian_kl_divergence(
                    mu_z_q, mu_z_p))

                ze_l = model.inference_network.sample_z(he_l)
                zg_l = model.generation_network.sample_z(hg_l)

                hg_next, cg_next, u_next = model.generation_network.forward_onestep(
                    hg_l, cg_l, u_l, ze_l, query_viewpoints, r)

                hg_l = hg_next
                cg_l = cg_next
                u_l = u_next
                he_l = he_next
                ce_l = ce_next

            generated_images = model.generation_network.sample_x(
                u_l, pixel_ln_var)
            generated_images = chainer.cuda.to_cpu(generated_images.data)
            generated_images = generated_images.transpose(0, 2, 3, 1)

            query_images = chainer.cuda.to_cpu(query_images)
            query_images = query_images.transpose(0, 2, 3, 1)

            if window.closed():
                exit()

            for batch_index in range(args.batch_size):
                axis = axes[batch_index * 2 + 0]
                image = query_images[batch_index]
                axis.update(
                    np.uint8(np.clip((image + 1.0) * 0.5 * 255, 0, 255)))

                axis = axes[batch_index * 2 + 1]
                image = generated_images[batch_index]
                axis.update(
                    np.uint8(np.clip((image + 1.0) * 0.5 * 255, 0, 255)))

            time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="../rooms_dataset")
    parser.add_argument("--snapshot-path", type=str, default="../snapshot")
    parser.add_argument("--batch-size", "-b", type=int, default=16)
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    args = parser.parse_args()
    main()
