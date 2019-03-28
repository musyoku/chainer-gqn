from .sampler import Sampler

class Iterator:
    def __init__(self, subset, batch_size, drop_last=True):
        self.sampler = Sampler(subset)
        self.drop_last = drop_last
        self.batch_size = batch_size

    def __len__(self):
        return len(self.sampler) // self.batch_size

    def __iter__(self):
        batch = []
        for index in self.sampler:
            batch.append(int(index))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch
