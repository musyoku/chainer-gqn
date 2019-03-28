import argparse
import math
import time
import sys
import os
import random

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import chainer
import chainer.functions as cf
import cupy
import numpy as np
from chainer.backends import cuda

sys.path.append(".")
import gqn
from gqn.preprocessing import preprocess_images, make_uint8
from hyperparams import HyperParameters
from model import Model


def to_gpu(array):
    if isinstance(array, np.ndarray):
        return cuda.to_gpu(array)
    return array


def main():
    try:
        os.mkdir(args.figure_directory)
    except:
        pass

    xp = np
    using_gpu = args.gpu_device >= 0
    if using_gpu:
        cuda.get_device(args.gpu_device).use()
        xp = cupy

    dataset = gqn.data.Dataset(args.dataset_path)

    hyperparams = HyperParameters(snapshot_directory=args.snapshot_path)
    model = Model(hyperparams, snapshot_directory=args.snapshot_path)
    if using_gpu:
        model.to_gpu()

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle("GQN")
    axis_observations = fig.add_subplot(1, 3, 1)
    axis_observations.axis("off")
    axis_observations.set_title("Observations")
    axis_ground_truth = fig.add_subplot(1, 3, 2)
    axis_ground_truth.axis("off")
    axis_ground_truth.set_title("Ground Truth")
    axis_reconstruction = fig.add_subplot(1, 3, 3)
    axis_reconstruction.axis("off")
    axis_reconstruction.set_title("Reconstruction")

    total_observations_per_scene = 2**2
    num_observations_per_column = int(math.sqrt(total_observations_per_scene))

    black_color = -0.5
    image_shape = (3, ) + hyperparams.image_size
    axis_observations_image = np.full(
        (3, num_observations_per_column * image_shape[1],
         num_observations_per_column * image_shape[2]),
        black_color,
        dtype=np.float32)
    file_number = 1

    with chainer.no_backprop_mode():
        for subset in dataset:
            iterator = gqn.data.Iterator(subset, batch_size=1)

            for data_indices in iterator:
                animation_frame_array = []
                axis_observations_image[...] = black_color

                observed_image_array = xp.full(
                    (total_observations_per_scene, ) + image_shape,
                    black_color,
                    dtype=np.float32)
                observed_viewpoint_array = xp.zeros(
                    (total_observations_per_scene, 7), dtype=np.float32)

                # shape: (batch, views, height, width, channels)
                # range: [-1, 1]
                images, viewpoints = subset[data_indices]

                # (batch, views, height, width, channels) -> (batch, views, channels, height, width)
                images = images.transpose((0, 1, 4, 2, 3)).astype(np.float32)
                images = preprocess_images(images)

                batch_index = 0

                query_index = total_observations_per_scene
                query_image = images[batch_index, query_index]
                query_viewpoint = to_gpu(
                    viewpoints[None, batch_index, query_index])

                axis_ground_truth.imshow(
                    make_uint8(query_image), interpolation="none")

                for observation_index in range(total_observations_per_scene):
                    observed_image = images[batch_index, observation_index]
                    observed_viewpoint = viewpoints[batch_index,
                                                    observation_index]

                    observed_image_array[observation_index] = to_gpu(
                        observed_image)
                    observed_viewpoint_array[observation_index] = to_gpu(
                        observed_viewpoint)

                    representation = model.compute_observation_representation(
                        observed_image_array[None, :observation_index + 1],
                        observed_viewpoint_array[None, :observation_index + 1])

                    representation = cf.broadcast_to(
                        representation, (1, ) + representation.shape[1:])

                    # Update figure
                    x_start = image_shape[1] * (
                        observation_index % num_observations_per_column)
                    x_end = x_start + image_shape[1]
                    y_start = image_shape[2] * (
                        observation_index // num_observations_per_column)
                    y_end = y_start + image_shape[2]
                    axis_observations_image[:, y_start:y_end, x_start:
                                            x_end] = observed_image

                    axis_observations.imshow(
                        make_uint8(axis_observations_image),
                        interpolation="none",
                        animated=True)

                    generated_images = model.generate_image(
                        query_viewpoint, representation)[0]

                    axis_reconstruction.imshow(
                        make_uint8(generated_images), interpolation="none")

                    plt.pause(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", "-dataset", type=str, required=True)
    parser.add_argument(
        "--snapshot-path", "-snapshot", type=str, required=True)
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    parser.add_argument(
        "--figure-directory", "-fig", type=str, default="figures")
    args = parser.parse_args()
    main()
