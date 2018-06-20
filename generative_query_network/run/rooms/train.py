import argparse
import sys
import os
import numpy as np
import cupy as xp
import chainer

sys.path.append(os.path.join("..", ".."))
import gqn
from hyper_parameters import HyperParameters
from model import Model


def main():
    dataset = gqn.data.Dataset(args.dataset_path)
    sampler = gqn.data.Sampler(dataset)
    iterator = gqn.data.Iterator(sampler, batch_size=args.batch_size)

    hyperparams = HyperParameters()
    model = Model(hyperparams)
    model.to_gpu()

    for indices in iterator:
        images, viewpoints = dataset[indices]

        # [batch, height, width, channels] -> [batch, channels, height, width]
        images = images.transpose(0, 3, 1, 2)
        images = chainer.cuda.to_gpu(images)
        viewpoints = chainer.cuda.to_gpu(viewpoints)

        image_size = images.shape[2:]

        r = model.representation_network.compute_r(images, viewpoints)

        hg_0 = xp.zeros(
            (
                args.batch_size,
                hyperparams.chrz_channels,
            ) + hyperparams.chrz_size,
            dtype="float32")
        cg_0 = xp.zeros(
            (
                args.batch_size,
                hyperparams.chrz_channels,
            ) + hyperparams.chrz_size,
            dtype="float32")
        u_0 = xp.zeros(
            (
                args.batch_size,
                hyperparams.generator_u_channels,
            ) + image_size,
            dtype="float32")
        he_0 = xp.zeros(
            (
                args.batch_size,
                hyperparams.chrz_channels,
            ) + hyperparams.chrz_size,
            dtype="float32")
        ce_0 = xp.zeros(
            (
                args.batch_size,
                hyperparams.chrz_channels,
            ) + hyperparams.chrz_size,
            dtype="float32")

        zg_l = model.generation_network.sample_z(hg_0)
        hg_l, cg_l, u_l = model.generation_network.forward_onestep(
            hg_0, cg_0, u_0, zg_l, viewpoints, r)
        x = model.generation_network.sample_x(u_l)

        he_l, ce_l = model.inference_network.forward_onestep(
            hg_0, he_0, ce_0, images, viewpoints, r)
        ze_l = model.inference_network.sample_z(he_l)
        hg_l, cg_l, u_l = model.generation_network.forward_onestep(
            hg_0, cg_0, u_0, ze_l, viewpoints, r)

        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="rooms_dataset")
    parser.add_argument("--batch-size", "-b", type=int, default=32)
    args = parser.parse_args()
    main()
