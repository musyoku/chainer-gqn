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
    meter_train = Meter()
    assert meter_train.load(args.snapshot_directory)

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
    dataset_test = Dataset(args.test_dataset_directory)

    #==============================================================================
    # Hyperparameters
    #==============================================================================
    hyperparams = HyperParameters()
    assert hyperparams.load(args.snapshot_directory)
    print(hyperparams, "\n")

    #==============================================================================
    # Model
    #==============================================================================
    model = Model(hyperparams)
    assert model.load(args.snapshot_directory, meter_train.epoch)
    if using_gpu:
        model.to_gpu()

    #==============================================================================
    # Pixel-variance annealing
    #==============================================================================
    variance_scheduler = PixelVarianceScheduler()
    assert variance_scheduler.load(args.snapshot_directory)
    print(variance_scheduler, "\n")

    #==============================================================================
    # Visualization
    #==============================================================================
    fig = plt.figure(figsize=(6, 3))
    axes_test = [
        fig.add_subplot(1, 2, 1),
        fig.add_subplot(1, 2, 2),
    ]
    axes_test[0].set_title("Validation Data")
    axes_test[0].axis("off")
    axes_test[1].set_title("Reconstruction")
    axes_test[1].axis("off")

    #==============================================================================
    # Algorithms
    #==============================================================================
    def encode_scene(images, viewpoints):
        # (batch, views, height, width, channels) -> (batch, views, channels, height, width)
        images = images.transpose((0, 1, 4, 2, 3)).astype(np.float32)

        # Sample number of views
        total_views = images.shape[1]
        num_views = random.choice(range(1, total_views + 1))

        print("num_views", num_views)

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
    # Test the model
    #==============================================================================
    random.seed(0)
    np.random.seed(0)
    meter = Meter()
    pixel_log_sigma = xp.full(
        (args.batch_size, 3) + hyperparams.image_size,
        math.log(variance_scheduler.standard_deviation),
        dtype="float32")

    with chainer.no_backprop_mode():
        for subset_index, subset in enumerate(dataset_test):
            iterator = Iterator(subset, batch_size=args.batch_size)
            for data_indices in iterator:
                images, viewpoints = subset[data_indices]

                norm = np.linalg.norm(viewpoints, axis=2)
                print(np.linalg.norm(viewpoints, axis=2))

                # Scene encoder
                representation, query_images, query_viewpoints = encode_scene(
                    images, viewpoints)


                # Compute empirical ELBO
                (z_t_param_array,
                 pixel_mean) = model.sample_z_and_x_params_from_posterior(
                     query_images, query_viewpoints, representation)
                (ELBO, bits_per_pixel, negative_log_likelihood,
                 kl_divergence) = estimate_ELBO(query_images, z_t_param_array,
                                                pixel_mean, pixel_log_sigma)
                mean_squared_error = cf.mean_squared_error(
                    query_images, pixel_mean)

                # Logging
                meter.update(
                    ELBO=float(ELBO.data),
                    bits_per_pixel=float(bits_per_pixel.data),
                    negative_log_likelihood=float(
                        negative_log_likelihood.data),
                    kl_divergence=float(kl_divergence.data),
                    mean_squared_error=float(mean_squared_error.data))

                axes_test[0].imshow(
                    make_uint8(query_images[0]), interpolation="none")
                axes_test[1].imshow(
                    make_uint8(pixel_mean.data[0]), interpolation="none")
                plt.pause(10)

            print("        {}".format(meter))


            if subset_index % 100 == 0:
                print("    Subset {}/{}:".format(
                    subset_index + 1,
                    len(dataset_test),
                ))
                print("        {}".format(meter))

    print("    Test:")
    print("        {} - done in {:.3f} min".format(
        meter,
        meter.elapsed_time,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dataset-directory", type=str, required=True)
    parser.add_argument("--snapshot-directory", type=str, required=True)
    parser.add_argument("--log-directory", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=196)
    parser.add_argument("--gpu-device", type=int, default=0)
    args = parser.parse_args()
    main()
