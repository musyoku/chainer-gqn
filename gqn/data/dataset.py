import os
import sys
import random
import numpy as np
from .subset import Subset


class Dataset():
    def __init__(self, directory):
        self.directory = directory
        self.current_subset_index = 0
        self.subset_filenames = []
        files = os.listdir(os.path.join(self.directory, "images"))
        for filename in files:
            if filename.endswith(".npy"):
                self.subset_filenames.append(filename)
        self.subset_filenames.sort()

    def __iter__(self):
        self.current_subset_index = 0
        random.shuffle(self.subset_filenames)
        return self

    def __next__(self):
        if self.current_subset_index >= len(self.subset_filenames):
            raise StopIteration
        subset = self.read(self.current_subset_index)
        self.current_subset_index += 1
        return subset

    def read(self, subset_index):
        filename = self.subset_filenames[self.current_subset_index]
        images_npy_path = os.path.join(self.directory, "images", filename)
        viewpoints_npy_path = os.path.join(self.directory, "viewpoints",
                                           filename)
        subset = Subset(images_npy_path, viewpoints_npy_path)
        return subset

    def __len__(self):
        return len(self.subset_filenames)
