import os
import random
import numpy as np
from .subset import Subset


class Dataset():
    def __init__(self, path):
        self.path = path
        self.current_subset_index = 0
        self.subset_filenames = []
        files = os.listdir(os.path.join(self.path, "images"))
        for filename in files:
            if filename.endswith(".npy"):
                self.subset_filenames.append(filename)

    def __iter__(self):
        self.current_subset_index = 0
        random.shuffle(self.subset_filenames)
        return self

    def __next__(self):
        if self.current_subset_index >= len(self.subset_filenames):
            raise StopIteration
        filename = self.subset_filenames[self.current_subset_index]
        images_npy_path = os.path.join(self.path, "images", filename)
        viewpoints_npy_path = os.path.join(self.path, "viewpoints", filename)
        subset = Subset(images_npy_path, viewpoints_npy_path)
        self.current_subset_index += 1
        return subset

    def __len__(self):
        return len(self.subset_filenames)
