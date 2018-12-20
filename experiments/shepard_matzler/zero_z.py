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


def generate_random_query_viewpoint(num_generation, xp):
    view_radius = 3
    eye = np.random.normal(size=3)
    eye = tuple(view_radius * (eye / np.linalg.norm(eye)))
    center = (0, 0, 0)
    yaw = gqn.math.yaw(eye, center)
    pitch = gqn.math.pitch(eye, center)
    query_viewpoints = xp.array(
        (eye[0], eye[1], eye[2], math.cos(yaw), math.sin(yaw), math.cos(pitch),
         math.sin(pitch)),
        dtype=np.float32)
    query_viewpoints = xp.broadcast_to(
        query_viewpoints, (num_generation, ) + query_viewpoints.shape)
    return query_viewpoints


def rotate_query_viewpoint(angle_rad, num_generation, xp):
    view_radius = 3
    eye = (view_radius * math.sin(angle_rad),
           view_radius * math.sin(angle_rad),
           view_radius * math.cos(angle_rad))
    center = (0, 0, 0)
    yaw = gqn.math.yaw(eye, center)
    pitch = gqn.math.pitch(eye, center)
    query_viewpoints = xp.array(
        (eye[0], eye[1], eye[2], math.cos(yaw), math.sin(yaw), math.cos(pitch),
         math.sin(pitch)),
        dtype=np.float32)
    query_viewpoints = xp.broadcast_to(
        query_viewpoints, (num_generation, ) + query_viewpoints.shape)
    return query_viewpoints


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
    fig = plt.figure(figsize=(10, 5))
    fig.suptitle("GQN")
    axis_observations = fig.add_subplot(1, 2, 1)
    axis_observations.axis("off")
    axis_observations.set_title("Observations")
    axis_generation = fig.add_subplot(1, 2, 2)
    axis_generation.axis("off")
    axis_generation.set_title("Generation")

    total_observations_per_scene = 2**2
    num_observations_per_column = int(math.sqrt(total_observations_per_scene))
    num_generation = 1
    total_frames_per_rotation = 48

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

                # Generate images without observations
                representation = xp.zeros(
                    (
                        num_generation,
                        hyperparams.representation_channels,
                    ) + (hyperparams.image_size[0] // 4,
                         hyperparams.image_size[1] // 4),
                    dtype=np.float32)

                angle_rad = 0
                for t in range(total_frames_per_rotation):
                    artist_array = [
                        axis_observations.imshow(
                            make_uint8(axis_observations_image),
                            interpolation="none",
                            animated=True)
                    ]

                    query_viewpoints = rotate_query_viewpoint(
                        angle_rad, num_generation, xp)
                    generated_image = model.generate_image_from_zero_z(
                        query_viewpoints, representation)[0]

                    artist_array.append(
                        axis_generation.imshow(
                            make_uint8(generated_image),
                            interpolation="none",
                            animated=True))

                    angle_rad += 2 * math.pi / total_frames_per_rotation
                    animation_frame_array.append(artist_array)

                # Generate images with observations
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
                        representation,
                        (num_generation, ) + representation.shape[1:])

                    # Update figure
                    x_start = image_shape[1] * (
                        observation_index % num_observations_per_column)
                    x_end = x_start + image_shape[1]
                    y_start = image_shape[2] * (
                        observation_index // num_observations_per_column)
                    y_end = y_start + image_shape[2]
                    axis_observations_image[:, y_start:y_end, x_start:
                                            x_end] = observed_image

                    angle_rad = 0
                    for t in range(total_frames_per_rotation):
                        artist_array = [
                            axis_observations.imshow(
                                make_uint8(axis_observations_image),
                                interpolation="none",
                                animated=True)
                        ]

                        query_viewpoints = rotate_query_viewpoint(
                            angle_rad, num_generation, xp)
                        generated_images = model.generate_image_from_zero_z(
                            query_viewpoints, representation)[0]

                        artist_array.append(
                            axis_generation.imshow(
                                make_uint8(generated_images),
                                interpolation="none",
                                animated=True))

                        angle_rad += 2 * math.pi / total_frames_per_rotation
                        animation_frame_array.append(artist_array)

                anim = animation.ArtistAnimation(
                    fig,
                    animation_frame_array,
                    interval=1 / 24,
                    blit=True,
                    repeat_delay=0)

                anim.save(
                    "{}/shepard_matzler_observations_{}.gif".format(
                        args.figure_directory, file_number),
                    writer="imagemagick")
                anim.save(
                    "{}/shepard_matzler_observations_{}.mp4".format(
                        args.figure_directory, file_number),
                    writer="ffmpeg",
                    fps=12)
                file_number += 1


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
