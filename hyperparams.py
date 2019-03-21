import json
import os

from tabulate import tabulate


class HyperParameters():
    def __init__(self, snapshot_directory=None):
        self.image_size = (64, 64)
        self.h_channels = 64
        self.z_channels = 3
        self.u_channels = 128
        self.r_channels = 256
        self.inference_share_core = False
        self.num_layers = 12
        self.generator_share_core = False
        self.initial_pixel_sigma = 2.0
        self.final_pixel_sigma = 0.7
        self.pixel_sigma_annealing_steps = 200000
        self.learning_rate_annealing_steps = 1600000
        self.representation_architecture = "tower"

        if snapshot_directory is not None:
            self.load(snapshot_directory)

    @property
    def filename(self):
        return "hyperparams.json"

    def load(self, snapshot_directory):
        json_path = os.path.join(snapshot_directory, self.filename)
        if os.path.exists(json_path) and os.path.isfile(json_path):
            with open(json_path, "r") as f:
                print("loading", json_path)
                obj = json.load(f)
                for (key, value) in obj.items():
                    if isinstance(value, list):
                        value = tuple(value)
                    setattr(self, key, value)

    def save(self, snapshot_directory):
        with open(os.path.join(snapshot_directory, self.filename), "w") as f:
            json.dump(self.__dict__, f, indent=4, sort_keys=True)

    def __str__(self):
        rows = []
        for key, value in self.__dict__.items():
            rows.append([key, value])
        return tabulate(rows, headers=["Hyperparameters", ""])
