import argparse
import math
import os
import random
import sys
import time

import chainer
import chainer.functions as cf
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from chainer.backends import cuda

sys.path.append(".")
import gqn
from gqn.preprocessing import make_uint8, preprocess_images
from hyperparams import HyperParameters
from functions import compute_yaw_and_pitch
from model import Model
from trainer.meter import Meter


def main():
    try:
        os.makedirs(args.figure_directory)
    except:
        pass

    xp = np
    using_gpu = args.gpu_device >= 0
    if using_gpu:
        cuda.get_device(args.gpu_device).use()
        xp = cp

    dataset = gqn.data.Dataset(args.dataset_directory)

    meter = Meter()
    assert meter.load(args.snapshot_directory)

    hyperparams = HyperParameters()
    assert hyperparams.load(args.snapshot_directory)

    model = Model(hyperparams)
    assert model.load(args.snapshot_directory, meter.epoch)

    if using_gpu:
        model.to_gpu()

    #==============================================================================
    # Visualization
    #==============================================================================
    plt.figure(figsize=(12, 16))

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

    #==============================================================================
    # Generating images
    #==============================================================================
    num_views_per_scene = 3
    num_yaw_pitch_steps = 10
    image_width, image_height = hyperparams.image_size
    prediction_images = make_uint8(
        np.full((num_yaw_pitch_steps * image_width,
                 num_yaw_pitch_steps * image_height, 3), 0))
    file_number = 1
    random.seed(0)
    np.random.seed(0)

    with chainer.no_backprop_mode():
        for subset in dataset:
            iterator = gqn.data.Iterator(subset, batch_size=1)

            for data_indices in iterator:
                # shape: (batch, views, height, width, channels)
                # range: [-1, 1]
                images, viewpoints = subset[data_indices]
                camera_distance = np.mean(
                    np.linalg.norm(viewpoints[:, :, :3], axis=2))

                # (batch, views, height, width, channels) -> (batch, views, channels, height, width)
                images = images.transpose((0, 1, 4, 2, 3)).astype(np.float32)
                images = preprocess_images(images)

                batch_index = 0

                #------------------------------------------------------------------------------
                # Observations
                #------------------------------------------------------------------------------
                total_views = images.shape[1]
                random_observation_view_indices = list(range(total_views))
                random.shuffle(random_observation_view_indices)
                random_observation_view_indices = random_observation_view_indices[:
                                                                                  num_views_per_scene]

                observed_images = images[:, random_observation_view_indices]
                observed_viewpoints = viewpoints[:,
                                                 random_observation_view_indices]
                representation = model.compute_observation_representation(
                    observed_images, observed_viewpoints)

                axis_observation_1.imshow(
                    make_uint8(observed_images[batch_index, 0]))
                axis_observation_2.imshow(
                    make_uint8(observed_images[batch_index, 1]))
                axis_observation_3.imshow(
                    make_uint8(observed_images[batch_index, 2]))

                y_angle_rad = math.pi / 2

                for pitch_loop in range(num_yaw_pitch_steps):
                    camera_y = math.sin(y_angle_rad)
                    x_angle_rad = math.pi

                    for yaw_loop in range(num_yaw_pitch_steps):
                        camera_direction = np.array([
                            math.sin(x_angle_rad), camera_y,
                            math.cos(x_angle_rad)
                        ])
                        camera_direction = camera_distance * camera_direction / np.linalg.norm(
                            camera_direction)
                        yaw, pitch = compute_yaw_and_pitch(camera_direction)

                        query_viewpoints = xp.array(
                            (
                                camera_direction[0],
                                camera_direction[1],
                                camera_direction[2],
                                math.cos(yaw),
                                math.sin(yaw),
                                math.cos(pitch),
                                math.sin(pitch),
                            ),
                            dtype=np.float32,
                        )
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

                        x_angle_rad -= 2 * math.pi / num_yaw_pitch_steps
                    y_angle_rad -= math.pi / num_yaw_pitch_steps

                axis_predictions.imshow(prediction_images)

                plt.savefig("{}/shepard_metzler_predictions_{}.png".format(
                    args.figure_directory, file_number))
                file_number += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--dataset-directory", type=str, required=True)
    parser.add_argument("--snapshot-directory", type=str, required=True)
    parser.add_argument("--figure-directory", type=str, required=True)
    args = parser.parse_args()
    main()
