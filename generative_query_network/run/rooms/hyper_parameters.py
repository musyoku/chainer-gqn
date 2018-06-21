class HyperParameters():
    def __init__(self):
        self.chrz_size = (16, 16)   # needs to be 1/4 of image_size
        self.channels_chz = 32
        self.channels_r = 256
        self.image_size = (64, 64)
        self.generator_total_timestep = 12
        self.generator_u_channels = 32
        self.generator_sigma_t = 1.0
        self.representation_architecture = "tower"