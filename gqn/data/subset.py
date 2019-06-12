import os
import h5py
import numpy as np


class Subset():
    def __init__(self, file_path):
        with h5py.File(file_path, "r") as f:
            self.images = f["images"][()]
            self.viewpoints = f["viewpoints"][()]
            assert self.images.shape[0] == self.viewpoints.shape[0]

    def __getitem__(self, indices):
        return self.images[indices], self.viewpoints[indices]

    def __len__(self):
        return self.images.shape[0]
