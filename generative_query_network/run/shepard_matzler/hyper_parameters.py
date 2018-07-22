class HyperParameters():
    def __init__(self):
        self.image_size = (64, 64)
        self.chrz_size = (16, 16)  # needs to be 1/4 of image_size
        self.channels_r = 256
        self.channels_chz = 64
        self.inference_channels_map_x = 32
        self.generator_generation_steps = 12
        self.generator_u_channels = 128
        self.pixel_sigma_i = 2.0
        self.pixel_sigma_f = 0.5
        self.pixel_n = 2 * 1e5
        self.representation_architecture = "tower"