import argparse
import math
import multiprocessing
import os
import random
import sys
import time

import chainer
import chainer.functions as cf
import chainermn
import cupy
import numpy as np
from chainer.backends import cuda

import gqn
from hyperparams import HyperParameters
from model import Model
from optimizer import AdamOptimizer, MomentumSGDOptimizer, RMSpropOptimizer
from scheduler import Scheduler


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


def main():
    ##############################################
    # To avoid OpenMPI bug
    multiprocessing.set_start_method("forkserver")
    p = multiprocessing.Process(target=print, args=("", ))
    p.start()
    p.join()
    ##############################################

    try:
        os.mkdir(args.snapshot_directory)
    except:
        pass

    comm = chainermn.create_communicator()
    device = comm.intra_rank
    print("device", device, "/", comm.size)
    cuda.get_device(device).use()
    xp = cupy

    dataset = gqn.data.Dataset(args.dataset_directory)

    hyperparams = HyperParameters()
    hyperparams.generator_share_core = args.generator_share_core
    hyperparams.generator_share_prior = args.generator_share_prior
    hyperparams.generator_generation_steps = args.generation_steps
    hyperparams.generator_downsampler_channels = args.generator_downsampler_channels
    hyperparams.generator_u_channels = args.u_channels
    hyperparams.generator_share_upsampler = args.generator_share_upsampler
    hyperparams.generator_lstm_peephole_enabled = args.generator_lstm_peephole_enabled
    hyperparams.inference_share_core = args.inference_share_core
    hyperparams.inference_share_posterior = args.inference_share_posterior
    hyperparams.inference_downsampler_channels = args.inference_downsampler_channels
    hyperparams.inference_lstm_peephole_enabled = args.inference_lstm_peephole_enabled
    hyperparams.chz_channels = args.chz_channels
    hyperparams.representation_channels = args.representation_channels
    hyperparams.pixel_n = args.pixel_n
    hyperparams.pixel_sigma_i = args.initial_pixel_sigma
    hyperparams.pixel_sigma_f = args.final_pixel_sigma
    if comm.rank == 0:
        hyperparams.save(args.snapshot_directory)
        print(hyperparams)

    model = Model(hyperparams, snapshot_directory=args.snapshot_directory)
    model.to_gpu()

    optimizer = AdamOptimizer(
        model.parameters,
        communicator=comm,
        mu_i=args.initial_lr,
        mu_f=args.final_lr)
    if comm.rank == 0:
        print(optimizer)

    scheduler = Scheduler()
    pixel_var = xp.full(
        (args.batch_size, 3) + hyperparams.image_size,
        scheduler.pixel_variance**2,
        dtype="float32")
    pixel_ln_var = xp.full(
        (args.batch_size, 3) + hyperparams.image_size,
        math.log(scheduler.pixel_variance**2),
        dtype="float32")
    num_pixels = hyperparams.image_size[0] * hyperparams.image_size[1] * 3

    random.seed(0)
    subset_indices = list(range(len(dataset.subset_filenames)))

    representation_shape = (
        args.batch_size,
        hyperparams.representation_channels) + hyperparams.chrz_size

    current_training_step = 0
    for iteration in range(args.training_iterations):
        mean_kld = 0
        mean_nll = 0
        mean_mse = 0
        total_batch = 0
        subset_size_per_gpu = len(subset_indices) // comm.size
        if len(subset_indices) % comm.size != 0:
            subset_size_per_gpu += 1
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
                images = images.transpose((0, 1, 4, 2, 3)).astype(np.float32)
                images = images / 255.0
                images += np.random.uniform(
                    0, 1.0 / 256.0, size=images.shape).astype(np.float32)

                total_views = images.shape[1]

                # Sample observations
                num_views = random.choice(range(total_views + 1))
                observation_view_indices = list(range(total_views))
                random.shuffle(observation_view_indices)
                observation_view_indices = observation_view_indices[:num_views]

                if current_training_step == 0 and num_views == 0:
                    num_views = 1  # avoid OpenMPI error

                if num_views > 0:
                    representation = model.compute_observation_representation(
                        images[:, observation_view_indices],
                        viewpoints[:, observation_view_indices])
                else:
                    representation = xp.zeros(
                        representation_shape, dtype="float32")
                    representation = chainer.Variable(representation)

                # Sample query
                query_index = random.choice(range(total_views))
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
                loss_sse = chainer.Variable(xp.zeros((), dtype=xp.float32))
                if scheduler.reconstruction_weight > 0:
                    for reconstrution_t in reconstrution_t_array:
                        loss_sse += cf.sum(
                            cf.squared_error(reconstrution_t, query_images))
                    loss_sse /= args.batch_size

                # Negative log-likelihood of generated image
                loss_nll = cf.sum(
                    gqn.nn.chainer.functions.gaussian_negative_log_likelihood(
                        query_images, mean_x, pixel_var, pixel_ln_var))

                # Calculate the average loss value
                loss_nll = loss_nll / args.batch_size + math.log(256.0)
                loss_kld = loss_kld / args.batch_size

                loss = (loss_nll / scheduler.pixel_variance) + (
                    loss_kld * scheduler.kl_weight) + (
                        loss_sse * scheduler.reconstruction_weight)

                model.cleargrads()
                loss.backward()
                optimizer.update(current_training_step)

                loss_nll = float(loss_nll.data) / num_pixels
                loss_kld = float(loss_kld.data)
                loss_mse = float(loss_sse.data) / num_pixels / (
                    hyperparams.generator_generation_steps - 1)

                if comm.rank == 0:
                    printr(
                        "Iteration {}: Subset {} / {}: Batch {} / {} - loss: nll_per_pixel: {:.6f} mse: {:.6f} kld: {:.6f} - lr: {:.4e} - pixel_variance: {:.6f} - kl_weight: {:.3f} - rec_weight: {:.3f} - step: {}".
                        format(iteration + 1, subset_loop + 1,
                               subset_size_per_gpu, batch_index + 1,
                               len(iterator), loss_nll, loss_mse, loss_kld,
                               optimizer.learning_rate,
                               scheduler.pixel_variance, scheduler.kl_weight,
                               scheduler.reconstruction_weight,
                               current_training_step))

                total_batch += 1
                current_training_step += comm.size
                mean_kld += loss_kld
                mean_nll += loss_nll
                mean_mse += loss_mse

                scheduler.step(current_training_step)
                pixel_var[...] = scheduler.pixel_variance**2
                pixel_ln_var[...] = math.log(scheduler.pixel_variance**2)

            if comm.rank == 0:
                model.serialize(args.snapshot_directory)

        if comm.rank == 0:
            elapsed_time = time.time() - start_time
            print(
                "\033[2KIteration {} - loss: nll_per_pixel: {:.6f} mse: {:.6f} kld: {:.6f} - lr: {:.4e} - pixel_variance: {:.6f} - step: {} - elapsed_time: {:.3f} min".
                format(iteration + 1, mean_nll / total_batch,
                       mean_mse / total_batch, mean_kld / total_batch,
                       optimizer.learning_rate, scheduler.pixel_variance,
                       current_training_step, elapsed_time / 60))
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
    parser.add_argument(
        "--inference-lstm-peephole-enabled",
        "-i-peephole",
        action="store_true")
    parser.add_argument(
        "--generator-lstm-peephole-enabled",
        "-g-peephole",
        action="store_true")
    parser.add_argument("--num-bits-x", "-bits", type=int, default=8)
    args = parser.parse_args()
    main()
