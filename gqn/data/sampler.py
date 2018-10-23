import numpy as np


class Sampler:
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __iter__(self):
        return iter(np.random.permutation(len(self.subset)))