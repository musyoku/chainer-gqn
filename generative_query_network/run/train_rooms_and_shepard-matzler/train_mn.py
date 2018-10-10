import argparse
import math
import os
import random
import sys
import time
import multiprocessing

import chainer
import chainermn
import chainer.functions as cf
import numpy as np
import cupy
from chainer.backends import cuda

sys.path.append("generative_query_network")
sys.path.append(os.path.join("..", ".."))
import gqn

from hyperparams import HyperParameters
from model import Model
from optimizer import Optimizer


def printr(string):
    sys.stdout.write(string)
    sys.stdout.write("\r")


def to_gpu(array):
    if isinstance(array, np.ndarray):
        return cuda.to_gpu(array)
    return array


def to_cpu(array):
    if isinstance(array, cupy.ndarray):
        return cuda.to_cpu(array)
    return array


def preprocess(image, num_bits_x):
    image = (image + 1.0) / 2.0 * 255
    num_bins_x = 2**num_bits_x
    if num_bits_x < 8:
        image = np.floor(image / (2**(8 - num_bits_x)))
    image = image / num_bins_x
    return image


def main():
    try:
        os.mkdir(args.snapshot_directory)
    except:
        pass

    comm = chainermn.create_communicator()
    device = comm.intra_rank
    print("device", device, "/", comm.size)
    cuda.get_device(device).use()
    xp = cupy

    # To avoid OpenMPI bug
    multiprocessing.set_start_method("forkserver")
    p = multiprocessing.Process(target=print, args=("", ))
    p.start()
    p.join()

    dataset = gqn.data.Dataset(args.dataset_directory)

    hyperparams = HyperParameters()
    hyperparams.generator_share_core = args.generator_share_core
    hyperparams.generator_share_prior = args.generator_share_prior
    hyperparams.generator_generation_steps = args.generation_steps
    hyperparams.generator_downsampler_channels = args.generator_downsampler_channels
    hyperparams.generator_u_channels = args.u_channels
    hyperparams.generator_share_upsampler = args.generator_share_upsampler
    hyperparams.inference_share_core = args.inference_share_core
    hyperparams.inference_share_posterior = args.inference_share_posterior
    hyperparams.inference_downsampler_channels = args.inference_downsampler_channels
    hyperparams.chz_channels = args.chz_channels
    hyperparams.representation_channels = args.representation_channels
    hyperparams.pixel_n = args.pixel_n
    hyperparams.pixel_sigma_i = args.initial_pixel_sigma
    hyperparams.pixel_sigma_f = args.final_pixel_sigma
    if comm.rank == 0:
        hyperparams.save(args.snapshot_directory)
        hyperparams.print()

    model = Model(hyperparams, snapshot_directory=args.snapshot_directory)
    model.to_gpu()

    optimizer = Optimizer(
        model.parameters,
        communicator=comm,
        mu_i=args.initial_lr,
        mu_f=args.final_lr)
    if comm.rank == 0:
        optimizer.print()

    sigma_t = hyperparams.pixel_sigma_i
    pixel_var = xp.full(
        (args.batch_size, 3) + hyperparams.image_size,
        sigma_t**2,
        dtype="float32")
    pixel_ln_var = xp.full(
        (args.batch_size, 3) + hyperparams.image_size,
        math.log(sigma_t**2),
        dtype="float32")
    num_pixels = hyperparams.image_size[0] * hyperparams.image_size[1] * 3
    num_bins_x = 2**args.num_bits_x

    random.seed(0)
    subset_indices = list(range(len(dataset.subset_filenames)))

    current_training_step = 0
    for iteration in range(args.training_iterations):
        mean_kld = 0
        mean_nll = 0
        mean_mse = 0
        total_batch = 0
        subset_size_per_gpu = len(subset_indices) // comm.size + 1
        start_time = time.time()

        for subset_loop in range(subset_size_per_gpu):
            random.shuffle(subset_indices)
            subset_index = subset_indices[comm.rank]
            subset = dataset.read(subset_index)
            iterator = gqn.data.Iterator(subset, batch_size=args.batch_size)

            for batch_index, data_indices in enumerate(iterator):
                # shape: (batch, views, height, width, channels)
                # range: [-1, 1]
                images, viewpoints = subset[data_indices]

                # (batch, views, height, width, channels) ->  (batch, views, channels, height, width)
                images = images.transpose((0, 1, 4, 2, 3))
                images = preprocess(images, args.num_bits_x)
                images += np.random.uniform(
                    0, 1.0 / num_bins_x, size=images.shape)

                total_views = images.shape[1]

                # sample number of views
                num_views = random.choice(range(total_views + 1))
                query_index = random.choice(range(total_views))

                if current_training_step == 0 and num_views == 0:
                    num_views = 1  # avoid OpenMPI error

                if num_views > 0:
                    representation = model.compute_observation_representation(
                        images[:, :num_views], viewpoints[:, :num_views])
                else:
                    representation = xp.zeros(
                        (args.batch_size, hyperparams.representation_channels)
                        + hyperparams.chrz_size,
                        dtype="float32")
                    representation = chainer.Variable(representation)

                query_images = images[:, query_index]
                query_viewpoints = viewpoints[:, query_index]

                # Transfer to gpu
                query_images = to_gpu(query_images)
                query_viewpoints = to_gpu(query_viewpoints)

                z_t_param_array, mean_x, reconstrution_t_array = model.sample_z_and_x_params_from_posterior(
                    query_images, query_viewpoints, representation)

                # Compute loss
                ## KL Divergence
                loss_kld = 0
                for params in z_t_param_array:
                    mean_z_q, ln_var_z_q, mean_z_p, ln_var_z_p = params
                    kld = gqn.nn.chainer.functions.gaussian_kl_divergence(
                        mean_z_q, ln_var_z_q, mean_z_p, ln_var_z_p)
                    loss_kld += cf.sum(kld)

                # Optional
                loss_sse = 0
                for reconstrution_t in reconstrution_t_array:
                    loss_sse += cf.sum(
                        cf.squared_error(reconstrution_t, query_images))

                # Negative log-likelihood of generated image
                negative_log_likelihood = gqn.nn.chainer.functions.gaussian_negative_log_likelihood(
                    query_images, mean_x, pixel_var, pixel_ln_var)
                loss_nll = cf.sum(negative_log_likelihood)

                # Calculate the average loss value
                loss_nll = loss_nll / args.batch_size + math.log(num_bins_x)
                loss_kld /= args.batch_size
                loss_sse /= args.batch_size
                if args.loss_alpha <= 0:
                    loss = loss_nll + loss_kld
                else:
                    loss = loss_nll + loss_kld + args.loss_alpha * loss_sse

                model.cleargrads()
                loss.backward()
                optimizer.update(current_training_step)

                if comm.rank == 0:
                    printr(
                        "Iteration {}: Subset {} / {}: Batch {} / {} - loss: nll_per_pixel: {:.6f} mse: {:.6f} kld: {:.6f} - lr: {:.4e} - sigma_t: {:.6f}".
                        format(
                            iteration + 1, subset_index + 1, len(dataset),
                            batch_index + 1, len(iterator),
                            float(loss_nll.data) / num_pixels,
                            float(loss_sse.data) / num_pixels /
                            (hyperparams.generator_generation_steps - 1),
                            float(loss_kld.data), optimizer.learning_rate,
                            sigma_t))

                sf = hyperparams.pixel_sigma_f
                si = hyperparams.pixel_sigma_i
                sigma_t = max(
                    sf + (si - sf) *
                    (1.0 - current_training_step / hyperparams.pixel_n), sf)

                pixel_var[...] = sigma_t**2
                pixel_ln_var[...] = math.log(sigma_t**2)

                total_batch += 1
                current_training_step += comm.size
                # current_training_step += 1
                mean_kld += float(loss_kld.data)
                mean_nll += float(loss_nll.data)
                mean_mse += float(loss_sse.data) / num_pixels / (
                    hyperparams.generator_generation_steps - 1)

            if comm.rank == 0:
                model.serialize(args.snapshot_directory)

        if comm.rank == 0:
            elapsed_time = time.time() - start_time
            print(
                "\033[2KIteration {} - loss: nll_per_pixel: {:.6f} mse: {:.6f} kld: {:.6f} - lr: {:.4e} - sigma_t: {:.6f} - step: {} - elapsed_time: {:.3f} min".
                format(iteration + 1, mean_nll / total_batch / num_pixels,
                       mean_mse / total_batch, mean_kld / total_batch,
                       optimizer.learning_rate, sigma_t, current_training_step,
                       elapsed_time / 60))
            model.serialize(args.snapshot_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-directory", "-dataset", type=str, default="dataset_train")
    parser.add_argument(
        "--snapshot-directory", "-snapshot", type=str, default="snapshot")
    parser.add_argument("--batch-size", "-b", type=int, default=36)
    parser.add_argument(
        "--training-iterations", "-iter", type=int, default=2 * 10**6)
    parser.add_argument("--generation-steps", "-gsteps", type=int, default=12)
    parser.add_argument(
        "--initial-lr", "-mu-i", type=float, default=5.0 * 1e-4)
    parser.add_argument("--final-lr", "-mu-f", type=float, default=5.0 * 1e-5)
    parser.add_argument(
        "--initial-pixel-sigma", "-ps-i", type=float, default=2.0)
    parser.add_argument(
        "--final-pixel-sigma", "-ps-f", type=float, default=0.7)
    parser.add_argument("--loss-alpha", "-lalpha", type=float, default=1.0)
    parser.add_argument("--pixel-n", "-pn", type=int, default=2 * 10**5)
    parser.add_argument("--chz-channels", "-cz", type=int, default=64)
    parser.add_argument("--u-channels", "-cu", type=int, default=64)
    parser.add_argument(
        "--representation-channels", "-cr", type=int, default=256)
    parser.add_argument(
        "--inference-downsampler-channels", "-cix", type=int, default=32)
    parser.add_argument(
        "--generator-downsampler-channels", "-cgx", type=int, default=32)
    parser.add_argument(
        "--generator-share-core", "-g-share-core", action="store_true")
    parser.add_argument(
        "--generator-share-prior", "-g-share-prior", action="store_true")
    parser.add_argument(
        "--generator-share-upsampler",
        "-g-share-upsampler",
        action="store_true")
    parser.add_argument(
        "--inference-share-core", "-i-share-core", action="store_true")
    parser.add_argument(
        "--inference-share-posterior",
        "-i-share-posterior",
        action="store_true")
    parser.add_argument("--num-bits-x", "-bits", type=int, default=8)
    args = parser.parse_args()
    main()
