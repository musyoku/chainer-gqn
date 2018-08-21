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


def generate_random_query_viewpoint(xp):
    rad = math.pi * 2 * np.random.uniform(0, 1, size=1)[0]
    eye = (3.0 * math.cos(rad), 0, 3.0 * math.sin(rad))
    center = (0.0, 0.5, 0.0)
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
        fov_rad=math.pi / 2.0,
        aspect_ratio=screen_size[0] / screen_size[1],
        z_near=0.1,
        z_far=10)

    figure = gqn.imgplot.figure()
    axis_observation = gqn.imgplot.image()
    axis_generation = gqn.imgplot.image()
    figure.add(axis_observation, 0, 0, 0.5, 1)
    figure.add(axis_generation, 0.5, 0, 0.5, 1)
    window = gqn.imgplot.window(figure, (1600, 800), "Viewpoint")
    window.show()

    raw_observed_image = np.zeros(screen_size + (3, ), dtype="uint32")
    renderer = gqn.three.Renderer(screen_size[0], screen_size[1])

    observed_images = xp.zeros((1, 3) + screen_size, dtype="float32")
    observed_viewpoint = xp.zeros((1, 7), dtype="float32")
    query_viewpoint = xp.zeros((1, 7), dtype="float32")

    with chainer.no_backprop_mode():
        while True:
            if window.closed():
                exit()

            scene, _ = gqn.environment.shepard_metzler.build_scene(
                num_blocks=random.choice([x for x in range(7, 8)]))
            renderer.set_scene(scene)

            rad = random.uniform(0, math.pi * 2)
            eye = (3.0 * math.cos(rad), 0, 3.0 * math.sin(rad))
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

            observed_images[0] = to_gpu(observe_image.transpose((2, 0, 1)))

            axis_observation.update(make_uint8(observed_images[0]))

            observed_viewpoint[0] = xp.array(
                (eye[0], eye[1], eye[2], math.cos(yaw), math.sin(yaw),
                 math.cos(pitch), math.sin(pitch)),
                dtype="float32")

            r = model.representation_network.compute_r(observed_images,
                                                       observed_viewpoint)
            num_samples = 50
            for _ in range(num_samples):
                if window.closed():
                    exit()

                yaw += np.random.normal(0, 0.05, size=1)[0]
                pitch += np.random.normal(0, 0.05, size=1)[0]
                query_viewpoint[0] = xp.array(
                    (eye[0], eye[1], eye[2], math.cos(yaw), math.sin(yaw),
                     math.cos(pitch), math.sin(pitch)),
                    dtype="float32")
                generated_image = model.generate_image(query_viewpoint, r, xp)
                axis_generation.update(make_uint8(generated_image[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot-path", "-snapshot", type=str, required=True)
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    args = parser.parse_args()
    main()
