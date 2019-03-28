import json
import os

from tabulate import tabulate
from gqn.json import JsonSerializable


class HyperParameters(JsonSerializable):
    def __init__(self):
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
        self.snapshot_filename = "hyperparams.json"

    def __str__(self):
        rows = []
        for key, value in self.__dict__.items():
            if key == "snapshot_filename":
                continue
            rows.append([key, value])
        return tabulate(rows, headers=["Hyperparameters", ""])
