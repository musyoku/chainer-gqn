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
        filename = self.subset_filenames[subset_index]
        images_npy_path = os.path.join(self.directory, "images", filename)
        viewpoints_npy_path = os.path.join(self.directory, "viewpoints", filename)
        subset = Subset(images_npy_path, viewpoints_npy_path)
        return subset

    def load_mean_and_std(self):
        dataset_mean = np.load(os.path.join(self.directory, "mean.npy"))
        dataset_std = np.load(os.path.join(self.directory, "std.npy"))
        return dataset_mean, dataset_std
        
        # try:
        #     dataset_mean = np.load(os.path.join(directory, "mean.npy"))
        #     dataset_std = np.load(os.path.join(directory, "std.npy"))
        #     return dataset_mean, dataset_std
        # except:
        #     total_size, subset_size = 0, 0
        #     dataset_mean = None
        #     dataset_var = None
        #     for subset_index, subset in enumerate(self):
        #         sys.stdout.write(
        #             "calculating the mean and variance of the dataset ... ({}/{})\r".
        #             format(subset_index, len(self)))
        #         subset_size = len(subset)
        #         new_total_size = total_size + subset_size
        #         co1 = total_size / new_total_size
        #         co2 = subset_size / new_total_size

        #         subset_mean = np.mean(subset.images, axis=(0, 1))
        #         subset_var = np.var(subset.images, axis=(0, 1))
        #         new_dataset_mean = subset_mean if dataset_mean is None else co1 * dataset_mean + co2 * subset_mean
        #         new_dataset_var = subset_var if dataset_var is None else co1 * (
        #             dataset_var + dataset_mean**2) + co2 * (
        #                 subset_var + subset_mean**2) - new_dataset_mean**2

        #         dataset_var = new_dataset_var
        #         dataset_mean = new_dataset_mean

        #         total_size += len(subset)
        #     dataset_std = np.sqrt(dataset_var) + 1e-12  # avoid division by zero

        #     print("\033[2K")

        #     np.save(os.path.join(directory, "mean.npy"), dataset_mean)
        #     np.save(os.path.join(directory, "std.npy"), dataset_std)

        #     return dataset_mean, dataset_std

    def __len__(self):
        return len(self.subset_filenames)
