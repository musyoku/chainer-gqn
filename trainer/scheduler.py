import json
import os

from tabulate import tabulate

from gqn.json import JsonSerializable


class PixelVarianceScheduler(JsonSerializable):
    def __init__(self,
                 sigma_start=2.0,
                 sigma_end=0.7,
                 final_num_updates=200000):
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.final_num_updates = final_num_updates
        self.snapshot_filename = "scheduler.json"
        self.update(0)

    def update(self, training_step):
        self.training_step = training_step
        steps = min(training_step, self.final_num_updates)
        self.standard_deviation = (self.sigma_start - self.sigma_end) * (
            1.0 - steps / self.final_num_updates) + self.sigma_end

    def __str__(self):
        rows = []
        for key, value in self.__dict__.items():
            if key == "snapshot_filename":
                continue
            rows.append([key, value])
        return tabulate(rows, headers=["Pixel-Variance Scheduler", ""])
