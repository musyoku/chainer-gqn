class Scheduler:
    def __init__(self):
        self.step(0)

    def step(self, num_updates):
        if num_updates < 5000:
            self.pixel_variance = 1.0
            self.kl_weight = 0
            self.reconstruction_weight = 1.0
            return
        self.pixel_variance = 0.7
        self.kl_weight = 1.0
        self.reconstruction_weight = 0.2
        return