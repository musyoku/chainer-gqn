class Iterator:
    def __init__(self, sampler, batch_size, drop_last=True):
        self.sampler = sampler
        self.drop_last = drop_last
        self.batch_size = batch_size

    def __iter__(self):
        batch = []
        for index in self.sampler:
            batch.append(int(index))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch
