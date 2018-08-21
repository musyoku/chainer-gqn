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


def main():
    dataset = gqn.data.Dataset(args.dataset_path)

    figure = gqn.imgplot.figure()
    axis = gqn.imgplot.image()
    figure.add(axis, 0, 0, 1, 1)
    window = gqn.imgplot.window(figure, (800, 800), "Dataset")
    window.show()

    with chainer.no_backprop_mode():
        for _, subset in enumerate(dataset):
            iterator = gqn.data.Iterator(subset, 1)

            for data_indices in iterator:
                images, viewpoints = subset[data_indices]

                for frames in images:
                    for image in frames:
                        image = (image + 1.0) * 127.5
                        axis.update(np.uint8(image))
                        time.sleep(0.1)

                if window.closed():
                    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", "-dataset", type=str, required=True)
    args = parser.parse_args()
    main()
