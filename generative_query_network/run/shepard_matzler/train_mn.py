import argparse
import math
import os
import random
import sys
import time

import chainer
import chainermn
import chainer.functions as cf
import cupy
from chainer.backends import cuda

sys.path.append("generative_query_network")
sys.path.append(os.path.join("..", ".."))
import gqn

from hyper_parameters import HyperParameters
from model import Model
from optimizer import Optimizer


def printr(string):
    sys.stdout.write(string)
    sys.stdout.write("\r")


def to_gpu(array):
    return cuda.to_gpu(array)


def to_cpu(array):
    return cuda.to_cpu(array)


def main():
    try:
        os.mkdir(args.snapshot_path)
    except:
        pass

    comm = chainermn.create_communicator()
    device = comm.intra_rank
    print("device", device, "/", comm.size)
    cuda.get_device(device).use()
    xp = cupy

    dataset = gqn.data.Dataset(args.dataset_path)

    hyperparams = HyperParameters()
    model = Model(hyperparams, hdf5_path=args.snapshot_path)
    model.to_gpu()

    optimizer = Optimizer(model.parameters, communicator=comm)

    sigma_t = hyperparams.pixel_sigma_i
    pixel_var = xp.full(
        (args.batch_size, 3) + hyperparams.image_size,
        sigma_t**2,
        dtype="float32")
    pixel_ln_var = xp.full(
        (args.batch_size, 3) + hyperparams.image_size,
        math.log(sigma_t**2),
        dtype="float32")

    random.seed(0)
    subset_indices = list(range(len(dataset.subset_filenames)))

    current_training_step = 0
    for iteration in range(args.training_steps):
        mean_kld = 0
        mean_nll = 0
        total_batch = 0
        subset_size_per_gpu = len(subset_indices) // comm.size
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

                # shape: (batch, views, height, width, channels)
                # range: [-1, 1]
                images = xp.asanyarray(images)
                image_size = images.shape[2:4]

                # sample number of views
                total_views = images.shape[1]
                num_views = random.choice(range(total_views))
                query_index = random.choice(range(total_views))

                if current_training_step == 0 and num_views == 0:
                    num_views = 1  # avoid OpenMPI error

                if num_views > 0:
                    observed_images = images[:, :num_views]
                    observed_viewpoints = viewpoints[:, :num_views]

                    # (batch, views, height, width, channels) -> (batch * views, height, width, channels)
                    observed_images = observed_images.reshape(
                        (args.batch_size * num_views, ) + images.shape[2:])
                    observed_viewpoints = observed_viewpoints.reshape(
                        (args.batch_size * num_views, ) +
                        observed_viewpoints.shape[2:])

                    # (batch * views, height, width, channels) -> (batch * views, channels, height, width)
                    observed_images = observed_images.transpose((0, 3, 1, 2))

                    # transfer to gpu
                    observed_images = to_gpu(observed_images)
                    observed_viewpoints = to_gpu(observed_viewpoints)

                    r = model.representation_network.compute_r(
                        observed_images, observed_viewpoints)

                    # (batch * views, channels, height, width) -> (batch, views, channels, height, width)
                    r = r.reshape((args.batch_size, num_views) + r.shape[1:])

                    # sum element-wise across views
                    r = cf.sum(r, axis=1)
                else:
                    r = xp.zeros(
                        (args.batch_size, hyperparams.channels_r) +
                        hyperparams.chrz_size,
                        dtype="float32")
                    r = chainer.Variable(r)

                query_images = images[:, query_index]
                query_viewpoints = viewpoints[:, query_index]

                # (batch * views, height, width, channels) -> (batch * views, channels, height, width)
                query_images = query_images.transpose((0, 3, 1, 2))

                # transfer to gpu
                query_images = to_gpu(query_images)
                query_viewpoints = to_gpu(query_viewpoints)

                h0_g, c0_g, u_0, h0_e, c0_e = model.generate_initial_state(
                    args.batch_size, xp)

                loss_kld = 0
                hl_e = h0_e
                cl_e = c0_e
                hl_g = h0_g
                cl_g = c0_g
                ul_e = u_0
                for l in range(model.generation_steps):
                    inference_core = model.get_inference_core(l)
                    generation_core = model.get_generation_core(l)

                    xq = model.inference_downsampler.downsample(query_images)

                    he_next, ce_next = inference_core.forward_onestep(
                        hl_g, hl_e, cl_e, xq, query_viewpoints, r)

                    mean_z_q = model.inference_posterior.compute_mean_z(hl_e)
                    ln_var_z_q = model.inference_posterior.compute_ln_var_z(
                        hl_e)
                    ze_l = cf.gaussian(mean_z_q, ln_var_z_q)

                    mean_z_p = model.generation_prior.compute_mean_z(hl_g)
                    ln_var_z_p = model.generation_prior.compute_ln_var_z(hl_g)

                    hg_next, cg_next, ue_next = generation_core.forward_onestep(
                        hl_g, cl_g, ul_e, ze_l, query_viewpoints, r)

                    kld = gqn.nn.chainer.functions.gaussian_kl_divergence(
                        mean_z_q, ln_var_z_q, mean_z_p, ln_var_z_p)

                    loss_kld += cf.sum(kld)

                    hl_g = hg_next
                    cl_g = cg_next
                    ul_e = ue_next
                    hl_e = he_next
                    cl_e = ce_next

                mean_x = model.generation_observation.compute_mean_x(ul_e)
                negative_log_likelihood = gqn.nn.chainer.functions.gaussian_negative_log_likelihood(
                    query_images, mean_x, pixel_var, pixel_ln_var)
                loss_nll = cf.sum(negative_log_likelihood)

                loss_nll /= args.batch_size
                loss_kld /= args.batch_size
                loss = loss_nll + loss_kld
                model.cleargrads()
                loss.backward()
                optimizer.update(current_training_step)

                if comm.rank == 0:
                    printr(
                        "Iteration {}: Subset {} / {}: Batch {} / {} - loss: nll: {:.3f} kld: {:.3f} - lr: {:.4e} - sigma_t: {:.6f}".
                        format(iteration + 1, subset_loop * comm.size + 1,
                               len(dataset), batch_index + 1,
                               len(subset) // args.batch_size,
                               float(loss_nll.data), float(loss_kld.data),
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
                mean_kld += float(loss_kld.data)
                mean_nll += float(loss_nll.data)

            if comm.rank == 0:
                model.serialize(args.snapshot_path)

        if comm.rank == 0:
            elapsed_time = time.time() - start_time
            print(
                "\033[2KIteration {} - loss: nll: {:.3f} kld: {:.3f} - lr: {:.4e} - sigma_t: {:.6f} - step: {} - elapsed_time: {:.3f} min".
                format(iteration + 1, mean_nll / total_batch,
                       mean_kld / total_batch, optimizer.learning_rate, sigma_t,
                       current_training_step, elapsed_time / 60))
            model.serialize(args.snapshot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="rooms_dataset")
    parser.add_argument("--subset-size", type=int, default=200)
    parser.add_argument("--snapshot-path", type=str, default="snapshot")
    parser.add_argument("--batch-size", "-b", type=int, default=36)
    parser.add_argument(
        "--with-visualization",
        "-visualize",
        action="store_true",
        default=False)
    parser.add_argument(
        "--training-steps", "-smax", type=int, default=2 * 10**6)
    args = parser.parse_args()
    main()
