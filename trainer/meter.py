import time
import json
import os
from gqn.json import JsonSerializable


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, value):
        self.value = value
        self.sum += value
        self.count += 1
        self.average = self.sum / self.count


class Meter(JsonSerializable):
    def __init__(self):
        self.ELBO = AverageMeter()
        self.bits_per_pixel = AverageMeter()
        self.negative_log_likelihood = AverageMeter()
        self.kl_divergence = AverageMeter()
        self.mean_squared_error = AverageMeter()
        self.start_time = time.time()
        self.epoch_start_time = time.time()
        self.num_updates = 0
        self.epoch = 0
        self.snapshot_filename = "meter.json"

    @property
    def elapsed_time(self):
        return (time.time() - self.start_time) / 60

    @property
    def epoch_elapsed_time(self):
        return (time.time() - self.epoch_start_time) / 60

    def update(self, ELBO, bits_per_pixel, negative_log_likelihood,
               kl_divergence, mean_squared_error):
        self.ELBO.update(ELBO)
        self.bits_per_pixel.update(bits_per_pixel)
        self.negative_log_likelihood.update(negative_log_likelihood)
        self.kl_divergence.update(kl_divergence)
        self.mean_squared_error.update(mean_squared_error)
        self.num_updates += 1

    def next_epoch(self):
        self.epoch += 1
        self.ELBO.reset()
        self.bits_per_pixel.reset()
        self.negative_log_likelihood.reset()
        self.kl_divergence.reset()
        self.mean_squared_error.reset()
        self.epoch_start_time = time.time()

    def __str__(self):
        return "ELBO: {} - bits/pixel: {} - MSE: {}".format(
            self.ELBO.average,
            self.bits_per_pixel.average,
            self.mean_squared_error.average,
        )

    def save(self, root_directory):
        path = os.path.join(root_directory, self.snapshot_filename)
        with open(path, "w") as f:
            json.dump({
                "num_updates": self.num_updates,
                "epoch": self.epoch,
            },
                      f,
                      indent=4,
                      sort_keys=True)

    def allreduce(self, communicator):
        meter = Meter()

        sum_ELBO = communicator.allreduce_obj(self.ELBO.average)
        meter.ELBO.average = sum_ELBO / communicator.size

        sum_bits_per_pixel = communicator.allreduce_obj(
            self.bits_per_pixel.average)
        meter.bits_per_pixel.average = sum_bits_per_pixel / communicator.size

        sum_mean_squared_error = communicator.allreduce_obj(
            self.mean_squared_error.average)
        meter.mean_squared_error.average = sum_mean_squared_error / communicator.size

        sum_kl_divergence = communicator.allreduce_obj(
            self.kl_divergence.average)
        meter.kl_divergence.average = sum_kl_divergence / communicator.size

        sum_negative_log_likelihood = communicator.allreduce_obj(
            self.negative_log_likelihood.average)
        meter.negative_log_likelihood.average = sum_negative_log_likelihood / communicator.size

        meter.start_time = self.start_time
        meter.epoch_start_time = self.epoch_start_time
        meter.num_updates = self.num_updates
        meter.epoch = self.epoch

        return meter