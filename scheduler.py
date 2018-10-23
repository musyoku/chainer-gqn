class Scheduler:
    def __init__(self, final_num_updates=100000):
        self.final_num_updates = final_num_updates
        self.step(0)

    def step(self, num_updates):
        num_updates = min(num_updates, self.final_num_updates)
        if num_updates < 10000:
            self.pixel_variance = 1.0
            self.kl_weight = 0
            self.reconstruction_weight = 1.0
            return
        sigma_start = 2.0
        sigma_end = 0.7
        self.pixel_variance = (sigma_start - sigma_end) * (
            1.0 - num_updates / self.final_num_updates) + sigma_end
        self.kl_weight = 1.0
        self.reconstruction_weight = 0
        return