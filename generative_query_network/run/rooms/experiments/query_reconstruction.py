import argparse
import sys
import os
import random
import math
import time
import numpy as np
import cupy
import chainer
import chainer.functions as cf
from chainer.backends import cuda

sys.path.append(os.path.join("..", "..", ".."))
import gqn

sys.path.append(os.path.join(".."))
from hyperparams import HyperParameters
from model import Model


def make_uint8(image, mean, std):
    if (image.shape[0] == 3):
        image = image.transpose(1, 2, 0)
    image = to_cpu(image)
    image = image * std + mean
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


def main():
    xp = np
    using_gpu = args.gpu_device >= 0
    if using_gpu:
        cuda.get_device(args.gpu_device).use()
        xp = cupy

    dataset = gqn.data.Dataset(args.dataset_path)
    sampler = gqn.data.Sampler(dataset)
    iterator = gqn.data.Iterator(sampler, batch_size=args.batch_size)

    hyperparams = HyperParameters(snapshot_directory=args.snapshot_path)
    model = Model(hyperparams, snapshot_directory=args.snapshot_path)
    if using_gpu:
        model.to_gpu()

    dataset_mean = np.load(os.path.join(args.snapshot_path, "mean.npy"))
    dataset_std = np.load(os.path.join(args.snapshot_path, "std.npy"))

    figure = gqn.imgplot.figure()
    axes = []
    sqrt_batch_size = int(math.sqrt(args.batch_size))
    axis_size = 1.0 / sqrt_batch_size
    for y in range(sqrt_batch_size):
        for x in range(sqrt_batch_size * 2):
            axis = gqn.imgplot.image()
            axes.append(axis)
            figure.add(axis, axis_size / 2 * x, axis_size * y, axis_size / 2,
                       axis_size)
    window = gqn.imgplot.window(figure, (1600, 800), "Reconstucted images")
    window.show()

    with chainer.no_backprop_mode():
        for _, subset in enumerate(dataset):
            iterator = gqn.data.Iterator(subset, batch_size=args.batch_size)

            for data_indices in iterator:
                # shape: (batch, views, height, width, channels)
                # range: [-1, 1]
                images, viewpoints = subset[data_indices]

                # preprocess
                images = (images - dataset_mean) / dataset_std

                # (batch, views, height, width, channels) -> (batch, views, channels, height, width)
                images = images.transpose((0, 1, 4, 2, 3))

                total_views = images.shape[1]

                # sample number of views
                num_views = random.choice(range(total_views))
                query_index = random.choice(range(total_views))

                if num_views > 0:
                    r = model.compute_observation_representation(
                        images[:, :num_views], viewpoints[:, :num_views])
                else:
                    r = np.zeros(
                        (args.batch_size, hyperparams.channels_r) +
                        hyperparams.chrz_size,
                        dtype="float32")
                    r = to_gpu(r)

                query_images = images[:, query_index]
                query_viewpoints = viewpoints[:, query_index]

                # transfer to gpu
                query_images = to_gpu(query_images)
                query_viewpoints = to_gpu(query_viewpoints)

                reconstructed_images = model.generate_image(
                    query_viewpoints, r, xp)

                if window.closed():
                    exit()

                for batch_index in range(args.batch_size):
                    axis = axes[batch_index * 2 + 0]
                    image = query_images[batch_index]
                    axis.update(make_uint8(image, dataset_mean, dataset_std))

                    axis = axes[batch_index * 2 + 1]
                    image = reconstructed_images[batch_index]
                    axis.update(make_uint8(image, dataset_mean, dataset_std))

                time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", "-dataset", type=str, required=True)
    parser.add_argument(
        "--snapshot-path", "-snapshot", type=str, required=True)
    parser.add_argument("--batch-size", "-b", type=int, default=16)
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    args = parser.parse_args()
    main()
