import argparse
import math
import time
import sys
import os
import random

import matplotlib.pyplot as plt
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

    fig = plt.figure(figsize=(12, 16))

    axis_observation_1 = plt.subplot2grid((4, 3), (0, 0))
    axis_observation_2 = plt.subplot2grid((4, 3), (0, 1))
    axis_observation_3 = plt.subplot2grid((4, 3), (0, 2))

    axis_predictions = plt.subplot2grid((4, 3), (1, 0), rowspan=3, colspan=3)

    axis_observation_1.axis("off")
    axis_observation_2.axis("off")
    axis_observation_3.axis("off")
    axis_predictions.set_xticks([], [])
    axis_predictions.set_yticks([], [])

    axis_observation_1.set_title("Observation 1", fontsize=22)
    axis_observation_2.set_title("Observation 2", fontsize=22)
    axis_observation_3.set_title("Observation 3", fontsize=22)

    axis_predictions.set_title("Neural Rendering", fontsize=22)
    axis_predictions.set_xlabel("Yaw", fontsize=22)
    axis_predictions.set_ylabel("Pitch", fontsize=22)

    num_views_per_scene = 3
    num_yaw_pitch_steps = 10
    image_width, image_height = hyperparams.image_size
    image_shape = (3, ) + hyperparams.image_size
    prediction_images = make_uint8(
        np.full((num_yaw_pitch_steps * image_width,
                 num_yaw_pitch_steps * image_height, 3), 0))
    file_number = 1

    with chainer.no_backprop_mode():
        for subset in dataset:
            iterator = gqn.data.Iterator(subset, batch_size=1)

            for data_indices in iterator:
                # shape: (batch, views, height, width, channels)
                # range: [-1, 1]
                images, viewpoints = subset[data_indices]

                # (batch, views, height, width, channels) -> (batch, views, channels, height, width)
                images = images.transpose((0, 1, 4, 2, 3)).astype(np.float32)
                images = preprocess_images(images)

                batch_index = 0

                total_views = images.shape[1]
                observation_view_indices = list(range(total_views))
                random.shuffle(observation_view_indices)
                observation_view_indices = observation_view_indices[:
                                                                    num_views_per_scene]

                observed_image_array = images[:, observation_view_indices]
                representation = model.compute_observation_representation(
                    observed_image_array,
                    viewpoints[:, observation_view_indices])

                axis_observation_1.imshow(
                    make_uint8(observed_image_array[batch_index, 0]))
                axis_observation_2.imshow(
                    make_uint8(observed_image_array[batch_index, 1]))
                axis_observation_3.imshow(
                    make_uint8(observed_image_array[batch_index, 2]))

                x_angle_rad = math.pi / 2
                for pitch_loop in range(num_yaw_pitch_steps):
                    y_angle_rad = math.pi
                    for yaw_loop in range(num_yaw_pitch_steps):
                        eye_norm = 3
                        eye_y = eye_norm * math.sin(x_angle_rad)
                        radius = math.cos(x_angle_rad)
                        eye = (radius * math.sin(y_angle_rad), eye_y,
                               radius * math.cos(y_angle_rad))
                        center = (0, 0, 0)
                        yaw = gqn.math.yaw(eye, center)
                        pitch = gqn.math.pitch(eye, center)
                        query_viewpoints = xp.array(
                            (eye[0], eye[1], eye[2], math.cos(yaw),
                             math.sin(yaw), math.cos(pitch), math.sin(pitch)),
                            dtype=np.float32)
                        query_viewpoints = xp.broadcast_to(
                            query_viewpoints, (1, ) + query_viewpoints.shape)

                        generated_images = model.generate_image(
                            query_viewpoints, representation)[0]

                        yi_start = pitch_loop * image_height
                        yi_end = (pitch_loop + 1) * image_height
                        xi_start = yaw_loop * image_width
                        xi_end = (yaw_loop + 1) * image_width
                        prediction_images[yi_start:yi_end, xi_start:
                                          xi_end] = make_uint8(
                                              generated_images)

                        y_angle_rad -= 2 * math.pi / num_yaw_pitch_steps
                    x_angle_rad -= math.pi / num_yaw_pitch_steps

                axis_predictions.imshow(prediction_images)

                plt.savefig("{}/shepard_matzler_predictions_{}.png".format(
                    args.figure_directory, file_number))
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