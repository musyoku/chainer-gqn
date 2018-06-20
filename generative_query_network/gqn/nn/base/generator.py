class Network:
    def generate_x(self, v, r):
        raise NotImplementedError

    def forward_onestep(self, prev_h, prev_c, prev_u, prev_z, v, r):
        raise NotImplementedError

    def sample_x(self, u):
        raise NotImplementedError

    def sample_z(self, h):
        raise NotImplementedError