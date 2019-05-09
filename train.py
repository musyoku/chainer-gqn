import argparse
import math
import os
import random
import sys

import chainer
import chainer.functions as cf
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from chainer.backends import cuda

import gqn
from gqn import to_device
from gqn.data import Dataset, Iterator
from gqn.preprocessing import make_uint8, preprocess_images
from hyperparams import HyperParameters
from model import Model
from trainer.dataframe import DataFrame
from trainer.meter import Meter
from trainer.optimizer import AdamOptimizer
from trainer.scheduler import PixelVarianceScheduler


def _mkdir(directory):
    try:
        os.makedirs(directory)
    except:
        pass


def main():
    _mkdir(args.snapshot_directory)
    _mkdir(args.log_directory)

    meter_train = Meter()
    meter_train.load(args.snapshot_directory)

    #==============================================================================
    # Selecting the GPU
    #==============================================================================
    xp = np
    gpu_device = args.gpu_device
    using_gpu = gpu_device >= 0
    if using_gpu:
        cuda.get_device(gpu_device).use()
        xp = cp

    #==============================================================================
    # Dataset
    #==============================================================================
    dataset_train = Dataset(args.train_dataset_directory)
    dataset_test = None
    if args.test_dataset_directory is not None:
        dataset_test = Dataset(args.test_dataset_directory)

    #==============================================================================
    # Hyperparameters
    #==============================================================================
    hyperparams = HyperParameters()
    hyperparams.num_layers = args.generation_steps
    hyperparams.generator_share_core = args.generator_share_core
    hyperparams.inference_share_core = args.inference_share_core
    hyperparams.h_channels = args.h_channels
    hyperparams.z_channels = args.z_channels
    hyperparams.u_channels = args.u_channels
    hyperparams.r_channels = args.r_channels
    hyperparams.image_size = (args.image_size, args.image_size)
    hyperparams.representation_architecture = args.representation_architecture
    hyperparams.pixel_sigma_annealing_steps = args.pixel_sigma_annealing_steps
    hyperparams.initial_pixel_sigma = args.initial_pixel_sigma
    hyperparams.final_pixel_sigma = args.final_pixel_sigma

    hyperparams.save(args.snapshot_directory)
    print(hyperparams, "\n")

    #==============================================================================
    # Model
    #==============================================================================
    model = Model(hyperparams)
    model.load(args.snapshot_directory, meter_train.epoch)
    if using_gpu:
        model.to_gpu()

    #==============================================================================
    # Pixel-variance annealing
    #==============================================================================
    variance_scheduler = PixelVarianceScheduler(
        sigma_start=args.initial_pixel_sigma,
        sigma_end=args.final_pixel_sigma,
        final_num_updates=args.pixel_sigma_annealing_steps)
    variance_scheduler.load(args.snapshot_directory)
    print(variance_scheduler, "\n")

    pixel_log_sigma = xp.full(
        (args.batch_size, 3) + hyperparams.image_size,
        math.log(variance_scheduler.standard_deviation),
        dtype="float32")

    #==============================================================================
    # Logging
    #==============================================================================
    csv = DataFrame()
    csv.load(args.log_directory)

    #==============================================================================
    # Optimizer
    #==============================================================================
    optimizer = AdamOptimizer(
        model.parameters,
        initial_lr=args.initial_lr,
        final_lr=args.final_lr,
        initial_training_step=variance_scheduler.training_step)
    print(optimizer, "\n")

    #==============================================================================
    # Visualization
    #==============================================================================
    fig = plt.figure(figsize=(9, 6))
    axes_train = [
        fig.add_subplot(2, 3, 1),
        fig.add_subplot(2, 3, 2),
        fig.add_subplot(2, 3, 3),
    ]
    axes_train[0].set_title("Training Data")
    axes_train[0].axis("off")
    axes_train[1].set_title("Reconstruction")
    axes_train[1].axis("off")
    axes_train[2].set_title("Generation")
    axes_train[2].axis("off")
    axes_test = [
        fig.add_subplot(2, 3, 4),
        fig.add_subplot(2, 3, 5),
        fig.add_subplot(2, 3, 6),
    ]
    axes_test[0].set_title("Validation Data")
    axes_test[0].axis("off")
    axes_test[1].set_title("Reconstruction")
    axes_test[1].axis("off")
    axes_test[2].set_title("Generation")
    axes_test[2].axis("off")

    #==============================================================================
    # Algorithms
    #==============================================================================
    def encode_scene(images, viewpoints):
        # (batch, views, height, width, channels) -> (batch, views, channels, height, width)
        images = images.transpose((0, 1, 4, 2, 3)).astype(np.float32)

        # Sample number of views
        total_views = images.shape[1]
        num_views = random.choice(range(1, total_views + 1))

        # Sample views
        observation_view_indices = list(range(total_views))
        random.shuffle(observation_view_indices)
        observation_view_indices = observation_view_indices[:num_views]

        observation_images = preprocess_images(
            images[:, observation_view_indices])

        observation_query = viewpoints[:, observation_view_indices]
        representation = model.compute_observation_representation(
            observation_images, observation_query)

        # Sample query view
        query_index = random.choice(range(total_views))
        query_images = preprocess_images(images[:, query_index])
        query_viewpoints = viewpoints[:, query_index]

        # Transfer to gpu if necessary
        query_images = to_device(query_images, gpu_device)
        query_viewpoints = to_device(query_viewpoints, gpu_device)

        return representation, query_images, query_viewpoints

    def estimate_ELBO(query_images, z_t_param_array, pixel_mean,
                      pixel_log_sigma):
        # KL Diverge, pixel_ln_varnce
        kl_divergence = 0
        for params_t in z_t_param_array:
            mean_z_q, ln_var_z_q, mean_z_p, ln_var_z_p = params_t
            normal_q = chainer.distributions.Normal(
                mean_z_q, log_scale=ln_var_z_q)
            normal_p = chainer.distributions.Normal(
                mean_z_p, log_scale=ln_var_z_p)
            kld_t = chainer.kl_divergence(normal_q, normal_p)
            kl_divergence += cf.sum(kld_t)
        kl_divergence = kl_divergence / args.batch_size

        # Negative log-likelihood of generated image
        batch_size = query_images.shape[0]
        num_pixels_per_batch = np.prod(query_images.shape[1:])
        normal = chainer.distributions.Normal(
            query_images, log_scale=pixel_log_sigma)

        log_px = cf.sum(normal.log_prob(pixel_mean)) / batch_size
        negative_log_likelihood = -log_px

        # Empirical ELBO
        ELBO = log_px - kl_divergence

        # https://arxiv.org/abs/1604.08772 Section.2
        # https://www.reddit.com/r/MachineLearning/comments/56m5o2/discussion_calculation_of_bitsdims/
        bits_per_pixel = -(ELBO / num_pixels_per_batch - np.log(256)) / np.log(
            2)

        return ELBO, bits_per_pixel, negative_log_likelihood, kl_divergence

    #==============================================================================
    # Training iterations
    #==============================================================================
    dataset_size = len(dataset_train)
    np.random.seed(0)
    cp.random.seed(0)

    for epoch in range(meter_train.epoch, args.epochs):
        print("Epoch {}/{}:".format(
            epoch + 1,
            args.epochs,
        ))
        meter_train.next_epoch()

        for subset_index, subset in enumerate(dataset_train):
            iterator = Iterator(subset, batch_size=args.batch_size)

            for batch_index, data_indices in enumerate(iterator):
                #------------------------------------------------------------------------------
                # Scene encoder
                #------------------------------------------------------------------------------
                # images.shape: (batch, views, height, width, channels)
                images, viewpoints = subset[data_indices]
                representation, query_images, query_viewpoints = encode_scene(
                    images, viewpoints)

                #------------------------------------------------------------------------------
                # Compute empirical ELBO
                #------------------------------------------------------------------------------
                # Compute distribution parameterws
                (z_t_param_array,
                 pixel_mean) = model.sample_z_and_x_params_from_posterior(
                     query_images, query_viewpoints, representation)

                # Compute ELBO
                (ELBO, bits_per_pixel, negative_log_likelihood,
                 kl_divergence) = estimate_ELBO(query_images, z_t_param_array,
                                                pixel_mean, pixel_log_sigma)

                #------------------------------------------------------------------------------
                # Update parameters
                #------------------------------------------------------------------------------
                loss = -ELBO
                model.cleargrads()
                loss.backward()
                optimizer.update(meter_train.num_updates)

                #------------------------------------------------------------------------------
                # Logging
                #------------------------------------------------------------------------------
                with chainer.no_backprop_mode():
                    mean_squared_error = cf.mean_squared_error(
                        query_images, pixel_mean)
                meter_train.update(
                    ELBO=float(ELBO.data),
                    bits_per_pixel=float(bits_per_pixel.data),
                    negative_log_likelihood=float(
                        negative_log_likelihood.data),
                    kl_divergence=float(kl_divergence.data),
                    mean_squared_error=float(mean_squared_error.data))

                #------------------------------------------------------------------------------
                # Annealing
                #------------------------------------------------------------------------------
                variance_scheduler.update(meter_train.num_updates)
                pixel_log_sigma[...] = math.log(
                    variance_scheduler.standard_deviation)

            if subset_index % 100 == 0:
                print("    Subset {}/{}:".format(
                    subset_index + 1,
                    dataset_size,
                ))
                print("        {}".format(meter_train))
                print("        lr: {} - sigma: {}".format(
                    optimizer.learning_rate,
                    variance_scheduler.standard_deviation))

        #------------------------------------------------------------------------------
        # Visualization
        #------------------------------------------------------------------------------
        if args.visualize:
            axes_train[0].imshow(
                make_uint8(query_images[0]), interpolation="none")
            axes_train[1].imshow(
                make_uint8(pixel_mean.data[0]), interpolation="none")

            with chainer.no_backprop_mode():
                generated_x = model.generate_image(query_viewpoints[None, 0],
                                                   representation[None, 0])
                axes_train[2].imshow(
                    make_uint8(generated_x[0]), interpolation="none")

        #------------------------------------------------------------------------------
        # Validation
        #------------------------------------------------------------------------------
        meter_test = None
        if dataset_test is not None:
            meter_test = Meter()
            batch_size_test = args.batch_size * 6
            pixel_log_sigma_test = xp.full(
                (batch_size_test, 3) + hyperparams.image_size,
                math.log(variance_scheduler.standard_deviation),
                dtype="float32")

            with chainer.no_backprop_mode():
                for subset in dataset_test:
                    iterator = Iterator(subset, batch_size=batch_size_test)
                    for data_indices in iterator:
                        images, viewpoints = subset[data_indices]

                        # Scene encoder
                        representation, query_images, query_viewpoints = encode_scene(
                            images, viewpoints)

                        # Compute empirical ELBO
                        (z_t_param_array, pixel_mean
                         ) = model.sample_z_and_x_params_from_posterior(
                             query_images, query_viewpoints, representation)
                        (ELBO, bits_per_pixel, negative_log_likelihood,
                         kl_divergence) = estimate_ELBO(
                             query_images, z_t_param_array, pixel_mean,
                             pixel_log_sigma_test)
                        mean_squared_error = cf.mean_squared_error(
                            query_images, pixel_mean)

                        # Logging
                        meter_test.update(
                            ELBO=float(ELBO.data),
                            bits_per_pixel=float(bits_per_pixel.data),
                            negative_log_likelihood=float(
                                negative_log_likelihood.data),
                            kl_divergence=float(kl_divergence.data),
                            mean_squared_error=float(mean_squared_error.data))

            print("    Test:")
            print("        {} - done in {:.3f} min".format(
                meter_test,
                meter_test.elapsed_time,
            ))

            if args.visualize:
                axes_test[0].imshow(
                    make_uint8(query_images[0]), interpolation="none")
                axes_test[1].imshow(
                    make_uint8(pixel_mean.data[0]), interpolation="none")

                with chainer.no_backprop_mode():
                    generated_x = model.generate_image(
                        query_viewpoints[None, 0], representation[None, 0])
                    axes_test[2].imshow(
                        make_uint8(generated_x[0]), interpolation="none")

        if args.visualize:
            plt.pause(1e-10)

        csv.append(epoch, meter_train, meter_test)

        #------------------------------------------------------------------------------
        # Snapshot
        #------------------------------------------------------------------------------
        model.save(args.snapshot_directory, epoch)
        variance_scheduler.save(args.snapshot_directory)
        meter_train.save(args.snapshot_directory)
        csv.save(args.log_directory)

        print("Epoch {} done in {:.3f} min".format(
            epoch + 1,
            meter_train.epoch_elapsed_time,
        ))
        print("    {}".format(meter_train))
        print("    lr: {} - sigma: {} - training_steps: {}".format(
            optimizer.learning_rate,
            variance_scheduler.standard_deviation,
            meter_train.num_updates,
        ))
        print("    Time elapsed: {:.3f} min".format(meter_train.elapsed_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset-directory", type=str, required=True)
    parser.add_argument("--test-dataset-directory", type=str, default=None)
    parser.add_argument("--snapshot-directory", type=str, default="snapshots")
    parser.add_argument("--log-directory", type=str, default="log")
    parser.add_argument("--batch-size", type=int, default=36)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--generation-steps", type=int, default=12)
    parser.add_argument("--initial-lr", type=float, default=1e-4)
    parser.add_argument("--final-lr", type=float, default=1e-5)
    parser.add_argument("--initial-pixel-sigma", type=float, default=2.0)
    parser.add_argument("--final-pixel-sigma", type=float, default=0.7)
    parser.add_argument(
        "--pixel-sigma-annealing-steps", type=int, default=160000)
    parser.add_argument("--h-channels", type=int, default=128)
    parser.add_argument("--z-channels", type=int, default=3)
    parser.add_argument("--u-channels", type=int, default=128)
    parser.add_argument("--r-channels", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument(
        "--representation-architecture",
        type=str,
        default="tower",
        choices=["tower", "pool"])
    parser.add_argument("--generator-share-core", action="store_true")
    parser.add_argument("--inference-share-core", action="store_true")
    parser.add_argument("--visualize", action="store_true", default=False)
    args = parser.parse_args()
    main()
