import argparse
import math
import os
import random
import sys
import time

import chainer
import chainer.functions as cf
import cupy
import matplotlib.pyplot as plt
import numpy as np
from chainer.backends import cuda

import gqn
from gqn import to_device
from gqn.preprocessing import preprocess_images, make_uint8
from hyperparams import HyperParameters
from model import Model
from optimizer import AdamOptimizer
from scheduler import Scheduler


def printr(string):
    sys.stdout.write(string)
    sys.stdout.write("\r")
    sys.stdout.flush()


def main():
    try:
        os.mkdir(args.snapshot_directory)
    except:
        pass

    np.random.seed(0)

    xp = np
    device_gpu = args.gpu_device
    device_cpu = -1
    using_gpu = device_gpu >= 0
    if using_gpu:
        cuda.get_device(args.gpu_device).use()
        xp = cupy

    dataset = gqn.data.Dataset(args.dataset_directory)

    hyperparams = HyperParameters()
    hyperparams.generator_share_core = args.generator_share_core
    hyperparams.generator_share_prior = args.generator_share_prior
    hyperparams.generator_generation_steps = args.generation_steps
    hyperparams.generator_share_upsampler = args.generator_share_upsampler
    hyperparams.inference_share_core = args.inference_share_core
    hyperparams.inference_share_posterior = args.inference_share_posterior
    hyperparams.h_channels = args.h_channels
    hyperparams.z_channels = args.z_channels
    hyperparams.u_channels = args.u_channels
    hyperparams.image_size = (args.image_size, args.image_size)
    hyperparams.representation_channels = args.representation_channels
    hyperparams.representation_architecture = args.representation_architecture
    hyperparams.pixel_n = args.pixel_n
    hyperparams.pixel_sigma_i = args.initial_pixel_variance
    hyperparams.pixel_sigma_f = args.final_pixel_variance
    hyperparams.save(args.snapshot_directory)
    print(hyperparams)

    model = Model(
        hyperparams,
        snapshot_directory=args.snapshot_directory,
        optimized=args.optimized)
    if using_gpu:
        model.to_gpu()

    scheduler = Scheduler(
        sigma_start=args.initial_pixel_variance,
        sigma_end=args.final_pixel_variance,
        final_num_updates=args.pixel_n,
        snapshot_directory=args.snapshot_directory)
    print(scheduler)

    optimizer = AdamOptimizer(
        model.parameters,
        mu_i=args.initial_lr,
        mu_f=args.final_lr,
        initial_training_step=scheduler.num_updates)
    print(optimizer)

    pixel_var = xp.full(
        (args.batch_size, 3) + hyperparams.image_size,
        scheduler.pixel_variance**2,
        dtype="float32")
    pixel_ln_var = xp.full(
        (args.batch_size, 3) + hyperparams.image_size,
        math.log(scheduler.pixel_variance**2),
        dtype="float32")

    representation_shape = (args.batch_size,
                            hyperparams.representation_channels,
                            args.image_size // 4, args.image_size // 4)

    fig = plt.figure(figsize=(9, 3))
    axis_data = fig.add_subplot(1, 3, 1)
    axis_data.set_title("Data")
    axis_data.axis("off")
    axis_reconstruction = fig.add_subplot(1, 3, 2)
    axis_reconstruction.set_title("Reconstruction")
    axis_reconstruction.axis("off")
    axis_generation = fig.add_subplot(1, 3, 3)
    axis_generation.set_title("Generation")
    axis_generation.axis("off")

    current_training_step = 0
    for iteration in range(args.training_iterations):
        mean_kld = 0
        mean_nll = 0
        mean_mse = 0
        mean_elbo = 0
        total_num_batch = 0
        start_time = time.time()

        for subset_index, subset in enumerate(dataset):
            iterator = gqn.data.Iterator(subset, batch_size=args.batch_size)

            for batch_index, data_indices in enumerate(iterator):
                # shape: (batch, views, height, width, channels)
                # range: [-1, 1]
                images, viewpoints = subset[data_indices]

                # (batch, views, height, width, channels) -> (batch, views, channels, height, width)
                images = images.transpose((0, 1, 4, 2, 3)).astype(np.float32)

                total_views = images.shape[1]

                # Sample number of views
                num_views = random.choice(range(1, total_views + 1))
                observation_view_indices = list(range(total_views))
                random.shuffle(observation_view_indices)
                observation_view_indices = observation_view_indices[:num_views]
                query_index = random.choice(range(total_views))

                if num_views > 0:
                    observation_images = preprocess_images(
                        images[:, observation_view_indices])
                    observation_query = viewpoints[:, observation_view_indices]
                    representation = model.compute_observation_representation(
                        observation_images, observation_query)
                else:
                    representation = xp.zeros(
                        representation_shape, dtype="float32")
                    representation = chainer.Variable(representation)

                # Sample query
                query_index = random.choice(range(total_views))
                query_images = preprocess_images(images[:, query_index])
                query_viewpoints = viewpoints[:, query_index]

                # Transfer to gpu if necessary
                query_images = to_device(query_images, device_gpu)
                query_viewpoints = to_device(query_viewpoints, device_gpu)

                z_t_param_array, mean_x = model.sample_z_and_x_params_from_posterior(
                    query_images, query_viewpoints, representation)

                # Compute loss
                ## KL Divergence
                loss_kld = 0
                for params in z_t_param_array:
                    mean_z_q, ln_var_z_q, mean_z_p, ln_var_z_p = params
                    kld = gqn.functions.gaussian_kl_divergence(
                        mean_z_q, ln_var_z_q, mean_z_p, ln_var_z_p)
                    loss_kld += cf.sum(kld)

                ## Negative log-likelihood of generated image
                loss_nll = cf.sum(
                    gqn.functions.gaussian_negative_log_likelihood(
                        query_images, mean_x, pixel_var, pixel_ln_var))

                # Calculate the average loss value
                loss_nll = loss_nll / args.batch_size
                loss_kld = loss_kld / args.batch_size

                loss = loss_nll / scheduler.pixel_variance + loss_kld

                model.cleargrads()
                loss.backward()
                optimizer.update(current_training_step)

                loss_nll = float(loss_nll.data) + math.log(256.0)
                loss_kld = float(loss_kld.data)

                elbo = -(loss_nll + loss_kld)

                loss_mse = float(
                    cf.mean_squared_error(query_images, mean_x).data)

                printr(
                    "Iteration {}: Subset {} / {}: Batch {} / {} - elbo: {:.2f} - loss: nll: {:.2f} mse: {:.6e} kld: {:.5f} - lr: {:.4e} - pixel_variance: {:.5f} - step: {}  ".
                    format(iteration + 1,
                           subset_index + 1, len(dataset), batch_index + 1,
                           len(iterator), elbo, loss_nll, loss_mse, loss_kld,
                           optimizer.learning_rate, scheduler.pixel_variance,
                           current_training_step))

                scheduler.step(iteration, current_training_step)
                pixel_var[...] = scheduler.pixel_variance**2
                pixel_ln_var[...] = math.log(scheduler.pixel_variance**2)

                total_num_batch += 1
                current_training_step += 1
                mean_kld += loss_kld
                mean_nll += loss_nll
                mean_mse += loss_mse
                mean_elbo += elbo

            model.serialize(args.snapshot_directory)

            # Visualize
            if args.with_visualization:
                axis_data.imshow(
                    make_uint8(query_images[0]), interpolation="none")
                axis_reconstruction.imshow(
                    make_uint8(mean_x.data[0]), interpolation="none")

                with chainer.no_backprop_mode():
                    generated_x = model.generate_image(
                        query_viewpoints[None, 0], representation[None, 0])
                    axis_generation.imshow(
                        make_uint8(generated_x[0]), interpolation="none")
                plt.pause(1e-8)

        elapsed_time = time.time() - start_time
        print(
            "\033[2KIteration {} - elbo: {:.2f} - loss: nll: {:.2f} mse: {:.6e} kld: {:.5f} - lr: {:.4e} - pixel_variance: {:.5f} - step: {} - time: {:.3f} min".
            format(iteration + 1, mean_elbo / total_num_batch,
                   mean_nll / total_num_batch, mean_mse / total_num_batch,
                   mean_kld / total_num_batch, optimizer.learning_rate,
                   scheduler.pixel_variance, current_training_step,
                   elapsed_time / 60))
        model.serialize(args.snapshot_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-directory", "-dataset", type=str, default="dataset_train")
    parser.add_argument(
        "--snapshot-directory", "-snapshot", type=str, default="snapshot")
    parser.add_argument("--batch-size", "-b", type=int, default=36)
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    parser.add_argument(
        "--training-iterations", "-iter", type=int, default=10000000)
    parser.add_argument("--generation-steps", "-gsteps", type=int, default=12)
    parser.add_argument("--initial-lr", "-mu-i", type=float, default=5e-4)
    parser.add_argument("--final-lr", "-mu-f", type=float, default=5e-5)
    parser.add_argument(
        "--initial-pixel-variance", "-ps-i", type=float, default=2.0)
    parser.add_argument(
        "--final-pixel-variance", "-ps-f", type=float, default=0.7)
    parser.add_argument("--pixel-n", "-pn", type=int, default=160000)
    parser.add_argument("--h-channels", "-ch", type=int, default=128)
    parser.add_argument("--z-channels", "-cz", type=int, default=3)
    parser.add_argument("--u-channels", "-cu", type=int, default=128)
    parser.add_argument("--image-size", "-is", type=int, default=64)
    parser.add_argument(
        "--representation-channels", "-cr", type=int, default=256)
    parser.add_argument(
        "--representation-architecture",
        "-r",
        type=str,
        default="tower",
        choices=["tower", "pool"])
    parser.add_argument(
        "--generator-share-core", "-g-share-core", action="store_true")
    parser.add_argument(
        "--generator-share-prior", "-g-share-prior", action="store_true")
    parser.add_argument(
        "--generator-share-upsampler",
        "-g-share-upsampler",
        action="store_true")
    parser.add_argument(
        "--inference-share-core", "-i-share-core", action="store_true")
    parser.add_argument(
        "--inference-share-posterior",
        "-i-share-posterior",
        action="store_true")
    parser.add_argument(
        "--with-visualization",
        "-visualize",
        action="store_true",
        default=False)
    parser.add_argument(
        "--optimized", "-optimized", action="store_true", default=False)
    args = parser.parse_args()
    main()
