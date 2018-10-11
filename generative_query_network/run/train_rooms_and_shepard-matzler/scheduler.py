class Scheduler:
    def __init__(self):
        self.pixel_variance = 0.2

    def step(self, num_updates):
        if num_updates < 100000:
            self.pixel_variance = 0.2
            return
        if num_updates < 200000:
            self.pixel_variance = 2.0
            return
        self.pixel_variance = 0.7
        return
