from tabulate import tabulate


class Scheduler:
    def __init__(self,
                 sigma_start,
                 sigma_end,
                 pretrain_steps=20000,
                 final_num_updates=200000):
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.pretrain_steps = pretrain_steps
        self.final_num_updates = final_num_updates
        self.step(0)

    def step(self, num_updates):
        num_updates = min(num_updates, self.final_num_updates)
        if num_updates < self.pretrain_steps:
            self.pixel_variance = 1.0
            self.kl_weight = 0
            self.reconstruction_weight = 1.0
            return
        self.pixel_variance = (self.sigma_start - self.sigma_end) * (
            1.0 - num_updates / self.final_num_updates) + self.sigma_end
        self.kl_weight = 1.0
        self.reconstruction_weight = 0
        return

    def __str__(self):
        rows = []
        for key, value in self.__dict__.items():
            rows.append([key, value])
        return tabulate(rows)