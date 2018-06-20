import numpy as np


class Sampler:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return iter(np.random.permutation(len(self.dataset)))