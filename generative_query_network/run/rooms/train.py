import argparse
import math
import os
import random
import sys

import chainer
import chainer.functions as cf
import cupy
import numpy as np
from chainer.backends import cuda

sys.path.append(os.path.join("..", ".."))
import gqn

from hyper_parameters import HyperParameters
from model import Model
from optimizer import Optimizer


def printr(string):
    sys.stdout.write(string)
    sys.stdout.write("\r")


def to_gpu(array):
    if args.gpu_device >= 0:
        return cuda.to_gpu(array)
    return array


def to_cpu(array):
    if args.gpu_device >= 0:
        return cuda.to_cpu(array)
    return array


def main():
    try:
        os.mkdir(args.snapshot_path)
    except:
        pass

    xp = np
    using_gpu = args.gpu_device >= 0
    if using_gpu:
        cuda.get_device(args.gpu_device).use()
        xp = cupy

    dataset = gqn.data.Dataset(args.dataset_path)

    hyperparams = HyperParameters()
    model = Model(hyperparams, hdf5_path=args.snapshot_path)
    if using_gpu:
        model.to_gpu()

    optimizer_all = Optimizer(model.all_parameters)
    optimizer_generation = Optimizer(model.generation_parameters)
    optimizer_inference = Optimizer(model.inference_parameters)

    figure = gqn.imgplot.Figure()
    axis1 = gqn.imgplot.ImageData(hyperparams.image_size[0],
                                  hyperparams.image_size[1], 3)
    axis2 = gqn.imgplot.ImageData(hyperparams.image_size[0],
                                  hyperparams.image_size[1], 3)
    axis3 = gqn.imgplot.ImageData(hyperparams.image_size[0],
                                  hyperparams.image_size[1], 3)
    figure.add(axis1, 0, 0, 1 / 3, 1)
    figure.add(axis2, 1 / 3, 0, 1 / 3, 1)
    figure.add(axis3, 2 / 3, 0, 1 / 3, 1)
    window = gqn.imgplot.Window(figure, (500 * 3, 500))
    window.show()

    sigma_t = hyperparams.pixel_sigma_i
    pixel_var = xp.full(
        (args.batch_size, 3) + hyperparams.image_size,
        sigma_t**2,
        dtype="float32")
    pixel_ln_var = xp.full(
        (args.batch_size, 3) + hyperparams.image_size,
        math.log(sigma_t**2),
        dtype="float32")

    current_training_step = 0
    for iteration in range(args.training_steps):
        mean_kld = 0
        mean_nll = 0
        total_batch = 0
        for subset_index, subset in enumerate(dataset):
            iterator = gqn.data.Iterator(subset, batch_size=args.batch_size)

            for batch_index, data_indices in enumerate(iterator):
                # shape: (batch, views, height, width, channels)
                # range: [-1, 1]
                images, viewpoints = subset[data_indices]

                image_size = images.shape[2:4]
                total_views = images.shape[1]

                # sample number of views
                num_views = random.choice(range(total_views))
                query_index = random.choice(range(total_views))

                if num_views > 0:
                    observed_images = images[:, :num_views]
                    observed_viewpoints = viewpoints[:, :num_views]

                    # (batch, views, height, width, channels) -> (batch * views, height, width, channels)
                    observed_images = observed_images.reshape(
                        (args.batch_size * num_views, ) +
                        observed_images.shape[2:])
                    observed_viewpoints = observed_viewpoints.reshape(
                        (args.batch_size * num_views, ) +
                        observed_viewpoints.shape[2:])

                    # (batch * views, height, width, channels) -> (batch * views, channels, height, width)
                    observed_images = observed_images.transpose((0, 3, 1, 2))

                    # transfer to gpu
                    observed_images = to_gpu(observed_images)
                    observed_viewpoints = to_gpu(observed_viewpoints)

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
                    r = chainer.Variable(to_gpu(r))

                query_images = images[:, query_index]
                query_viewpoints = viewpoints[:, query_index]

                # (batch * views, height, width, channels) -> (batch * views, channels, height, width)
                query_images = query_images.transpose((0, 3, 1, 2))

                # transfer to gpu
                query_images = to_gpu(query_images)
                query_viewpoints = to_gpu(query_viewpoints)

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

                # Inference
                loss_kld = 0
                he_l = he_0
                ce_l = ce_0
                hg_l = hg_0
                cg_l = cg_0
                ue_l = u_0
                for l in range(hyperparams.generator_total_timestep):
                    he_next, ce_next = model.inference_network.forward_onestep(
                        hg_l, he_l, ce_l, query_images, query_viewpoints, r)

                    mean_z_q = model.inference_network.compute_mean_z(he_l)
                    ln_var_z_q = model.inference_network.compute_ln_var_z(he_l)
                    ze_l = cf.gaussian(mean_z_q, ln_var_z_q)

                    mean_z_p = model.generation_network.compute_mean_z(hg_l)
                    ln_var_z_p = model.generation_network.compute_ln_var_z(
                        he_l)

                    hg_next, cg_next, ue_next = model.generation_network.forward_onestep(
                        hg_l, cg_l, ue_l, ze_l, query_viewpoints, r)

                    kld = gqn.nn.chainer.functions.gaussian_kl_divergence(
                        mean_z_q, ln_var_z_q, mean_z_p, ln_var_z_p)

                    loss_kld += cf.sum(kld)

                    hg_l = hg_next
                    cg_l = cg_next
                    ue_l = ue_next
                    he_l = he_next
                    ce_l = ce_next

                mean_x = model.generation_network.compute_mean_x(ue_l)
                negative_log_likelihood = gqn.nn.chainer.functions.gaussian_negative_log_likelihood(
                    query_images, mean_x, pixel_var, pixel_ln_var)
                loss_nll = cf.sum(negative_log_likelihood)

                loss_nll /= args.batch_size
                loss_kld /= args.batch_size
                loss = loss_nll + loss_kld
                model.cleargrads()
                loss.backward()
                optimizer_all.update(current_training_step)

                if window.closed() is False:
                    axis1.update(
                        np.uint8(
                            (to_cpu(query_images[0].transpose(1, 2, 0)) + 1) *
                            0.5 * 255))

                    # Generation
                    with chainer.using_config("train",
                                              False), chainer.using_config(
                                                  "enable_backprop", False):
                        hg_l = hg_0
                        cg_l = cg_0
                        ug_l = u_0
                        for l in range(hyperparams.generator_total_timestep):
                            zg_l = model.generation_network.sample_z(hg_l)
                            hg_next, cg_next, ug_next = model.generation_network.forward_onestep(
                                hg_l, cg_l, ug_l, zg_l, query_viewpoints, r)

                            hg_l = hg_next
                            cg_l = cg_next
                            ug_l = ug_next

                    x = model.generation_network.sample_x(ug_l, pixel_ln_var)
                    axis2.update(
                        np.uint8(
                            np.clip((to_cpu(x.data[0].transpose(1, 2, 0)) + 1)
                                    * 0.5 * 255, 0, 255)))
                    x = model.generation_network.sample_x(ue_l, pixel_ln_var)
                    axis3.update(
                        np.uint8(
                            np.clip((to_cpu(x.data[0].transpose(1, 2, 0)) + 1)
                                    * 0.5 * 255, 0, 255)))

                # hg_l = hg_0
                # cg_l = cg_0
                # ug_l = u_0
                # for l in range(hyperparams.generator_total_timestep):
                #     zg_l = model.generation_network.sample_z(hg_l)
                #     hg_next, cg_next, ug_next = model.generation_network.forward_onestep(
                #         hg_l, cg_l, ug_l, zg_l, query_viewpoints, r)

                #     hg_l = hg_next
                #     cg_l = cg_next
                #     ug_l = ug_next

                # mean_x = model.generation_network.compute_mean_x(ug_l)
                # negative_log_likelihood = gqn.nn.chainer.functions.gaussian_negative_log_likelihood(
                #     query_images, mean_x, pixel_var, pixel_ln_var)

                # loss_nll = cf.mean(negative_log_likelihood)

                # model.cleargrads()
                # loss_nll.backward()
                # optimizer_generation.step(current_training_step)

                # if window.closed() is False:
                #     x = model.generation_network.sample_x(ug_l, pixel_ln_var)
                #     axis1.update(
                #         np.uint8((to_cpu(query_images[0].transpose(
                #             1, 2, 0)) + 1) * 0.5 * 255))
                #     axis2.update(
                #         np.uint8(
                #             np.clip((to_cpu(x.data[0].transpose(
                #                 1, 2, 0)) + 1) * 0.5 * 255, 0, 255)))

                printr(
                    "Iteration {}: Subset {} / {}: Batch {} / {} - loss: nll: {:.3f} kld: {:.3f} - lr: {:.4e} - sigma_t: {:.6f}".
                    format(iteration + 1,
                           subset_index + 1, len(dataset), batch_index + 1,
                           len(iterator), float(loss_nll.data),
                           float(loss_kld.data), optimizer_all.optimizer.alpha,
                           sigma_t))

                sf = hyperparams.pixel_sigma_f
                si = hyperparams.pixel_sigma_i
                sigma_t = max(
                    sf + (si - sf) *
                    (1.0 - current_training_step / hyperparams.pixel_n), sf)

                pixel_var[...] = sigma_t**2
                pixel_ln_var[...] = math.log(sigma_t**2)

                total_batch += 1
                current_training_step += 1
                mean_kld += float(loss_kld.data)
                mean_nll += float(loss_nll.data)

            model.serialize(args.snapshot_path)

        print(
            "\033[2KIteration {} - loss: nll: {:.3f} kld: {:.3f} - lr: {:.4e} - sigma_t: {:.6f} - step: {}".
            format(iteration + 1, mean_nll / total_batch,
                   mean_kld / total_batch, optimizer_all.optimizer.alpha,
                   sigma_t, current_training_step))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="rooms_dataset")
    parser.add_argument("--snapshot-path", type=str, default="snapshot")
    parser.add_argument("--batch-size", "-b", type=int, default=36)
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    parser.add_argument(
        "--training-steps", "-smax", type=int, default=2 * 10**6)
    args = parser.parse_args()
    main()
