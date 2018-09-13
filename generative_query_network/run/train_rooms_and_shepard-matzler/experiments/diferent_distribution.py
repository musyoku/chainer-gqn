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
from hyperparams import HyperParameters
from model import Model


def generate_blocks(block_positions,
                    color_hue_range=(0, 1),
                    color_saturation_range=(0.75, 1)):
    num_blocks = len(block_positions)
    ret = []
    center_of_gravity = [0, 0, 0]
    for position in block_positions:
        obj, _ = gqn.environment.objects.create_object(
            "cube",
            color=gqn.color.random_color(
                alpha=1,
                hue_range=color_hue_range,
                saturation_range=color_saturation_range))
        shift = 1
        location = (shift * position[0], shift * position[1],
                    shift * position[2])
        ret.append((obj, location))
        center_of_gravity[0] += location[0]
        center_of_gravity[1] += location[1]
        center_of_gravity[2] += location[2]
    center_of_gravity[0] /= num_blocks
    center_of_gravity[1] /= num_blocks
    center_of_gravity[2] /= num_blocks
    return ret, center_of_gravity


def build_scene(block_positions,
                object_color_hue_range=(0, 1),
                object_color_saturation_range=(0.75, 1)):
    scene = gqn.three.Scene()
    blocks, center_of_gravity = generate_blocks(
        block_positions,
        color_hue_range=object_color_hue_range,
        color_saturation_range=object_color_saturation_range)
    objects = []
    for block in blocks:
        obj, location = block
        position = (location[0] - center_of_gravity[0],
                    location[1] - center_of_gravity[1],
                    location[2] - center_of_gravity[2])
        scene.add(obj, position=position)
        objects.append(obj)
    return scene, objects


def make_uint8(image):
    if (image.shape[0] == 3):
        image = image.transpose(1, 2, 0)
    image = to_cpu(image)
    image = (image + 1) * 0.5
    return np.uint8(np.clip(image * 255, 0, 255))


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
    eye = (6.0 * math.cos(rad), 6.0 * math.sin(math.pi / 4),
           6.0 * math.sin(rad))
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

    hyperparams = HyperParameters(snapshot_directory=args.snapshot_path)
    model = Model(hyperparams, snapshot_directory=args.snapshot_path)
    if using_gpu:
        model.to_gpu()

    screen_size = hyperparams.image_size
    camera = gqn.three.PerspectiveCamera(
        eye=(3, 1, 0),
        center=(0, 0, 0),
        up=(0, 1, 0),
        fov_rad=math.pi / 3.0,
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

    window = gqn.imgplot.window(figure, (1600, 800), "Viewpoint")
    window.show()

    raw_observed_image = np.zeros(screen_size + (3, ), dtype="uint32")
    renderer = gqn.three.Renderer(screen_size[0], screen_size[1])

    observed_images = xp.zeros(
        (args.num_views_per_scene, 3) + hyperparams.image_size,
        dtype="float32")
    observed_viewpoints = xp.zeros(
        (args.num_views_per_scene, 7), dtype="float32")

    block_positions_array = [
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, -1, 1],
            [0, -1, 1],
            [-1, -1, 1],
            [-1, 0, 1],
            [-1, 1, 1],
            [0, 1, 1],
            [1, 1, 0],
            [1, 0, 0],
            [1, -1, 0],
            [0, -1, 0],
            [-1, -1, 0],
            [-1, 0, 0],
            [-1, 1, 0],
            [0, 1, 0],
            [1, 1, -1],
            [1, 0, -1],
            [1, -1, -1],
            [0, -1, -1],
            [-1, -1, -1],
            [-1, 0, -1],
            [-1, 1, -1],
            [0, 1, -1],
        ],
        [(1.5, 1.5, 0), (-1.5, 1.5, 0), (1.5, -1.5, 0), (-1.5, -1.5, 0)],
        [(1, -1, 0), (0, 0, 0), (2, -2, 0), (-1, 1, 0), (-2, 2, 0)],
        [(1.5, 1.5, 0)],
    ]

    with chainer.no_backprop_mode():
        while True:
            if window.closed():
                exit()
            for block_positions in block_positions_array:
                scene, _ = build_scene(block_positions)
                renderer.set_scene(scene)

                images = np.zeros(
                    (
                        1,
                        args.num_views_per_scene,
                    ) + screen_size + (3, ),
                    dtype=np.float32)
                viewpoints = np.zeros(
                    (1, args.num_views_per_scene, 7), dtype=np.float32)

                for render_index in range(args.num_views_per_scene):
                    eye = np.random.normal(size=3)
                    eye = tuple(6.0 * (eye / np.linalg.norm(eye)))
                    center = (0, 0, 0)
                    yaw = gqn.math.yaw(eye, center)
                    pitch = gqn.math.pitch(eye, center)
                    camera.look_at(
                        eye=eye,
                        center=center,
                        up=(0.0, 1.0, 0.0),
                    )
                    renderer.render(camera, raw_observed_image)

                    # [0, 255] -> [-1, 1]
                    observe_image = (raw_observed_image / 255.0 - 0.5) * 2.0

                    images[0, render_index] = observe_image
                    viewpoints[0, render_index] = (eye[0], eye[1], eye[2],
                                                   math.cos(yaw),
                                                   math.sin(yaw),
                                                   math.cos(pitch),
                                                   math.sin(pitch))

                # (batch, views, height, width, channels) -> (batch, views, channels, height, width)
                images = images.transpose((0, 1, 4, 2, 3))

                # generate images without observations
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
                    generated_images = model.generate_image(
                        query_viewpoints, r, xp)

                    for m in range(args.num_generation):
                        if window.closed():
                            exit()
                        image = make_uint8(generated_images[m])
                        axis = axes_generations[m]
                        axis.update(image)

                # generate images with observations
                for n in range(args.num_views_per_scene):
                    observed_image = images[0, n]
                    observed_viewpoint = viewpoints[0, n]

                    observed_images[n] = to_gpu(observed_image)
                    observed_viewpoints[n] = to_gpu(observed_viewpoint)

                    r = model.compute_observation_representation(
                        observed_images[None, :n + 1],
                        observed_viewpoints[None, :n + 1])

                    r = cf.broadcast_to(r,
                                        (args.num_generation, ) + r.shape[1:])

                    axis = axes_observations[n]
                    axis.update(make_uint8(observed_image))

                    total_frames = 50
                    for tick in range(total_frames):
                        if window.closed():
                            exit()
                        query_viewpoints = generate_random_query_viewpoint(
                            tick / total_frames, xp)
                        generated_images = model.generate_image(
                            query_viewpoints, r, xp)

                        for m in range(args.num_generation):
                            if window.closed():
                                exit()
                            axis = axes_generations[m]
                            axis.update(make_uint8(generated_images[m]))

                black_image = make_uint8(np.full_like(observed_image, -1))
                for axis in axes_observations:
                    axis.update(black_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot-path", "-snapshot", type=str, required=True)
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    parser.add_argument("--num-views-per-scene", "-k", type=int, default=4)
    parser.add_argument("--num-generation", "-g", type=int, default=4)
    args = parser.parse_args()
    main()
