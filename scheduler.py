import json
import os

from tabulate import tabulate


class Scheduler:
    def __init__(self,
                 sigma_start,
                 sigma_end,
                 final_num_updates=160000,
                 snapshot_directory=None):
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.final_num_updates = final_num_updates
        self.step(1, 0)

        if snapshot_directory is not None:
            json_path = os.path.join(snapshot_directory, self.filename)
            if os.path.exists(json_path) and os.path.isfile(json_path):
                with open(json_path, "r") as f:
                    print("loading", json_path)
                    obj = json.load(f)
                    for (key, value) in obj.items():
                        if isinstance(value, list):
                            value = tuple(value)
                        setattr(self, key, value)

    def step(self, iteration, num_updates):
        self.iteration = iteration
        self.num_updates = num_updates
        steps = min(num_updates, self.final_num_updates)
        self.pixel_variance = (self.sigma_start - self.sigma_end) * (
            1.0 - steps / self.final_num_updates) + self.sigma_end

    def __str__(self):
        rows = []
        for key, value in self.__dict__.items():
            rows.append([key, value])
        return tabulate(rows)

    @property
    def filename(self):
        return "scheduler.json"

    def save(self, snapshot_directory):
        with open(os.path.join(snapshot_directory, self.filename), "w") as f:
            json.dump(self.__dict__, f, indent=4, sort_keys=True)
