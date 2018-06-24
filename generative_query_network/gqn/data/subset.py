import os
import numpy as np


class Subset():
    def __init__(self, image_npy_path, viewpoints_npy_path):
        self.images = np.load(image_npy_path)
        self.viewpoints = np.load(viewpoints_npy_path)
        assert self.images.shape[0] == self.viewpoints.shape[0]

    def __getitem__(self, indices):
        return self.images[indices], self.viewpoints[indices]

    def __len__(self):
        return self.images.shape[0]
