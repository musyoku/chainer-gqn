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
    p = multiprocessing.Process(target=print, args=("",))
    p.start()
    p.join()

    dataset = gqn.data.Dataset(args.dataset_directory)

    hyperparams = HyperParameters()
    hyperparams.generator_share_core = args.generator_share_core
    hyperparams.generator_share_prior = args.generator_share_prior
    hyperparams.generator_generation_steps = args.generation_steps
    hyperparams.inference_share_core = args.inference_share_core
    hyperparams.inference_share_posterior = args.inference_share_posterior
    hyperparams.channels_chz = args.channels_chz
    hyperparams.generator_channels_u = args.channels_u
    hyperparams.inference_channels_map_x = args.channels_map_x
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
                    r = model.compute_observation_representation(
                        images[:, :num_views], viewpoints[:, :num_views])
                else:
                    r = xp.zeros(
                        (args.batch_size, hyperparams.channels_r) +
                        hyperparams.chrz_size,
                        dtype="float32")
                    r = chainer.Variable(r)

                query_images = images[:, query_index]
                query_viewpoints = viewpoints[:, query_index]

                # transfer to gpu
                query_images = to_gpu(query_images)
                query_viewpoints = to_gpu(query_viewpoints)

                h0_gen, c0_gen, u_0, h0_enc, c0_enc = model.generate_initial_state(
                    args.batch_size, xp)

                loss_kld = 0

                hl_enc = h0_enc
                cl_enc = c0_enc
                hl_gen = h0_gen
                cl_gen = c0_gen
                ul_enc = u_0

                xq = model.inference_downsampler.downsample(query_images)

                for l in range(model.generation_steps):
                    inference_core = model.get_inference_core(l)
                    inference_posterior = model.get_inference_posterior(l)
                    generation_core = model.get_generation_core(l)
                    generation_piror = model.get_generation_prior(l)

                    h_next_enc, c_next_enc = inference_core.forward_onestep(
                        hl_gen, hl_enc, cl_enc, xq, query_viewpoints, r)

                    mean_z_q = inference_posterior.compute_mean_z(hl_enc)
                    ln_var_z_q = inference_posterior.compute_ln_var_z(hl_enc)
                    ze_l = cf.gaussian(mean_z_q, ln_var_z_q)

                    mean_z_p = generation_piror.compute_mean_z(hl_gen)
                    ln_var_z_p = generation_piror.compute_ln_var_z(hl_gen)

                    h_next_gen, c_next_gen, u_next_enc = generation_core.forward_onestep(
                        hl_gen, cl_gen, ul_enc, ze_l, query_viewpoints, r)

                    kld = gqn.nn.chainer.functions.gaussian_kl_divergence(
                        mean_z_q, ln_var_z_q, mean_z_p, ln_var_z_p)

                    loss_kld += cf.sum(kld)

                    hl_gen = h_next_gen
                    cl_gen = c_next_gen
                    ul_enc = u_next_enc
                    hl_enc = h_next_enc
                    cl_enc = c_next_enc

                mean_x = model.generation_observation.compute_mean_x(ul_enc)
                negative_log_likelihood = gqn.nn.chainer.functions.gaussian_negative_log_likelihood(
                    query_images, mean_x, pixel_var, pixel_ln_var)
                loss_mse = cf.mean_squared_error(mean_x, query_images)
                loss_nll = cf.sum(negative_log_likelihood)
                # loss_nll = loss_mse * mean_x.size / sigma_t

                loss_nll /= args.batch_size
                loss_kld /= args.batch_size
                loss = loss_nll + loss_kld

                model.cleargrads()
                loss.backward()
                optimizer.update(current_training_step)

                if comm.rank == 0:
                    printr(
                        "Iteration {}: Subset {} / {}: Batch {} / {} - loss: nll_per_pixel: {:.6f} mse: {:.6f} kld: {:.6f} - lr: {:.4e} - sigma_t: {:.6f}".
                        format(iteration + 1, subset_loop * comm.size + 1,
                               len(dataset), batch_index + 1,
                               len(subset) // args.batch_size,
                               float(loss_nll.data) / num_pixels,
                               float(loss_mse.data), float(loss_kld.data),
                               optimizer.learning_rate, sigma_t))

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
                mean_mse += float(loss_mse.data)

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
    parser.add_argument("--pixel-n", "-pn", type=int, default=2 * 10**5)
    parser.add_argument("--channels-chz", "-cz", type=int, default=64)
    parser.add_argument("--channels-u", "-cu", type=int, default=128)
    parser.add_argument("--channels-map-x", "-cx", type=int, default=64)
    parser.add_argument(
        "--generator-share-core", "-g-share-core", action="store_true")
    parser.add_argument(
        "--generator-share-prior", "-g-share-prior", action="store_true")
    parser.add_argument(
        "--inference-share-core", "-i-share-core", action="store_true")
    parser.add_argument(
        "--inference-share-posterior",
        "-i-share-posterior",
        action="store_true")
    parser.add_argument("--num-bits-x", "-bits", type=int, default=8)
    args = parser.parse_args()
    main()
