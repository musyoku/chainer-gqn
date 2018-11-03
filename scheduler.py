from tabulate import tabulate


class Scheduler:
    def __init__(self, sigma_start, sigma_end, final_num_updates=160000):
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.final_num_updates = final_num_updates
        self.step(0)

    def step(self, num_updates):
        num_updates = min(num_updates, self.final_num_updates)
        self.pixel_variance = (self.sigma_start - self.sigma_end) * (
            1.0 - num_updates / self.final_num_updates) + self.sigma_end

    def __str__(self):
        rows = []
        for key, value in self.__dict__.items():
            rows.append([key, value])
        return tabulate(rows)