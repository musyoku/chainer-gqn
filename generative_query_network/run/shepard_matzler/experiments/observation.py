import argparse
import math
import time
import sys
import os
import random

import chainer
import chainer.functions as cf
import cupy
import numpy as np
from chainer.backends import cuda

sys.path.append(os.path.join("..", "..", ".."))
import gqn

sys.path.append(os.path.join(".."))
from hyper_parameters import HyperParameters
from model import Model


def make_uint8(array):
    if (array.shape[2] == 3):
        return np.uint8(np.clip((to_cpu(array) + 1) * 0.5 * 255, 0, 255))
    return np.uint8(
        np.clip((to_cpu(array.transpose(1, 2, 0)) + 1) * 0.5 * 255, 0, 255))


def to_gpu(array):
    if args.gpu_device >= 0:
        return cuda.to_gpu(array)
    return array


def to_cpu(array):
    if args.gpu_device >= 0:
        return cuda.to_cpu(array)
    return array


def generate_random_query_viewpoint(ratio, xp):
    rad = math.pi * 2 * ratio
    eye = (3.0 * math.cos(rad), 0, 3.0 * math.sin(rad))
    center = (0, 0, 0)
    yaw = gqn.math.yaw(eye, center)
    pitch = gqn.math.pitch(eye, center)
    query_viewpoints = xp.array(
        (eye[0], eye[1], eye[2], math.cos(yaw), math.sin(yaw), math.cos(pitch),
         math.sin(pitch)),
        dtype="float32")
    query_viewpoints = xp.broadcast_to(
        query_viewpoints, (args.num_generation, ) + query_viewpoints.shape)
    return query_viewpoints


def main():
    xp = np
    using_gpu = args.gpu_device >= 0
    if using_gpu:
        cuda.get_device(args.gpu_device).use()
        xp = cupy

    hyperparams = HyperParameters()
    model = Model(hyperparams, hdf5_path=args.snapshot_path)
    if using_gpu:
        model.to_gpu()

    screen_size = hyperparams.image_size
    camera = gqn.three.PerspectiveCamera(
        eye=(3, 1, 0),
        center=(0, 0, 0),
        up=(0, 1, 0),
        fov_rad=math.pi / 2.0,
        aspect_ratio=screen_size[0] / screen_size[1],
        z_near=0.1,
        z_far=10)

    figure = gqn.imgplot.figure()
    axes_observations = []
    axes_generations = []
    sqrt_n = math.sqrt(args.num_views_per_scene)
    axis_width = 0.5 / sqrt_n
    axis_height = 1.0 / sqrt_n
    for n in range(args.num_views_per_scene):
        axis = gqn.imgplot.image()
        x = n % sqrt_n
        y = n // sqrt_n
        figure.add(axis, x * axis_width, y * axis_height, axis_width,
                   axis_height)
        axes_observations.append(axis)
    sqrt_n = math.sqrt(args.num_generation)
    axis_width = 0.5 / sqrt_n
    axis_height = 1.0 / sqrt_n
    for n in range(args.num_generation):
        axis = gqn.imgplot.image()
        x = n % sqrt_n
        y = n // sqrt_n
        figure.add(axis, x * axis_width + 0.5, y * axis_height, axis_width,
                   axis_height)
        axes_generations.append(axis)

    window = gqn.imgplot.window(figure, (1600, 800), "Dataset")
    window.show()

    raw_observed_images = np.zeros(screen_size + (3, ), dtype="uint32")
    renderer = gqn.three.Renderer(screen_size[0], screen_size[1])

    observed_images = xp.zeros(
        (args.num_views_per_scene, 3) + screen_size, dtype="float32")
    observed_viewpoints = xp.zeros(
        (args.num_views_per_scene, 7), dtype="float32")

    with chainer.no_backprop_mode():
        while True:
            if window.closed():
                exit()

            scene, _ = gqn.environment.shepard_metzler.build_scene(
                num_blocks=random.choice([x for x in range(7, 8)]))
            renderer.set_scene(scene)

            # Generate images without observations
            r = xp.zeros(
                (
                    args.num_generation,
                    hyperparams.channels_r,
                ) + hyperparams.chrz_size,
                dtype="float32")
            total_frames = 50
            for tick in range(total_frames):
                if window.closed():
                    exit()
                query_viewpoints = generate_random_query_viewpoint(
                    tick / total_frames, xp)
                generated_images = to_cpu(
                    model.generate_image(query_viewpoints, r, xp))

                for m in range(args.num_generation):
                    if window.closed():
                        exit()
                    image = make_uint8(generated_images[m])
                    axis = axes_generations[m]
                    axis.update(image)

            for n in range(args.num_views_per_scene):
                if window.closed():
                    exit()
                rad = random.uniform(0, math.pi * 2)
                eye = (3.0 * math.cos(rad), 3.0 * math.sin(rad),
                       3.0 * math.sin(rad))
                center = (0, 0, 0)
                yaw = gqn.math.yaw(eye, center)
                pitch = gqn.math.pitch(eye, center)
                camera.look_at(
                    eye=eye,
                    center=center,
                    up=(0.0, 1.0, 0.0),
                )
                renderer.render(camera, raw_observed_images)

                # [0, 255] -> [-1, 1]
                observed_images[n] = to_gpu((raw_observed_images.transpose(
                    (2, 0, 1)) / 255 - 0.5) * 2.0)

                observed_viewpoints[n] = xp.array(
                    (eye[0], eye[1], eye[2], math.cos(yaw), math.sin(yaw),
                     math.cos(pitch), math.sin(pitch)),
                    dtype="float32")
                r = model.representation_network.compute_r(
                    observed_images[:n + 1], observed_viewpoints[:n + 1])

                # (batch * views, channels, height, width) -> (batch, views, channels, height, width)
                r = r.reshape((1, n + 1) + r.shape[1:])

                # sum element-wise across views
                r = cf.sum(r, axis=1)
                r = cf.broadcast_to(r, (args.num_generation, ) + r.shape[1:])

                axis = axes_observations[n]
                axis.update(np.uint8(raw_observed_images))

                total_frames = 50
                for tick in range(total_frames):
                    if window.closed():
                        exit()
                    query_viewpoints = generate_random_query_viewpoint(
                        tick / total_frames, xp)
                    generated_images = to_cpu(
                        model.generate_image(query_viewpoints, r, xp))

                    for m in range(args.num_generation):
                        if window.closed():
                            exit()
                        image = make_uint8(generated_images[m])
                        axis = axes_generations[m]
                        axis.update(image)

            raw_observed_images[...] = 0
            for n in range(args.num_views_per_scene):
                axis = axes_observations[n]
                axis.update(np.uint8(raw_observed_images))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-views-per-scene", "-k", type=int, default=9)
    parser.add_argument("--num-generation", "-g", type=int, default=4)
    parser.add_argument(
        "--snapshot-path", "-snapshot", type=str, required=True)
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    args = parser.parse_args()
    main()
