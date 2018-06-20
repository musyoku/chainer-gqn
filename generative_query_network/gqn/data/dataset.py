import os
import numpy as np


class Dataset():
    def __init__(self, path):
        self.path = path
        self.images = None
        self.viewpoints = None
        self.load_images()
        self.load_viewpoints()

    def load_images(self):
        files = os.listdir(os.path.join(self.path, "images"))
        for filename in files:
            filepath = os.path.join(self.path, "images", filename)
            images = np.load(filepath)
            self.images = images if self.images is None else np.concatenate(
                (self.images, images), axis=0)

    def load_viewpoints(self):
        files = os.listdir(os.path.join(self.path, "viewpoints"))
        for filename in files:
            filepath = os.path.join(self.path, "viewpoints", filename)
            viewpoints = np.load(filepath)
            self.viewpoints = viewpoints if self.viewpoints is None else np.concatenate(
                (self.viewpoints, viewpoints), axis=0)

    def __getitem__(self, indices):
        return self.images[indices], self.viewpoints[indices]

    def __len__(self):
        return self.images.shape[0]
