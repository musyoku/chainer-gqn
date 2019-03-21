import json
import os

from tabulate import tabulate


class PixelVarianceScheduler:
    def __init__(self, sigma_start, sigma_end, final_num_updates=200000):
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.final_num_updates = final_num_updates
        self.filename = "scheduler.json"
        self.update(0)

    def update(self, training_step):
        self.training_step = training_step
        steps = min(training_step, self.final_num_updates)
        self.standard_deviation = (self.sigma_start - self.sigma_end) * (
            1.0 - steps / self.final_num_updates) + self.sigma_end

    def __str__(self):
        rows = []
        for key, value in self.__dict__.items():
            rows.append([key, value])
        return tabulate(rows, headers=["Pixel-Variance Scheduler", ""])

    def load(self, path):
        if os.path.exists(path) and os.path.isfile(path):
            with open(path, "r") as f:
                print("loading", path)
                obj = json.load(f)
                for (key, value) in obj.items():
                    if isinstance(value, list):
                        value = tuple(value)
                    setattr(self, key, value)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4, sort_keys=True)
