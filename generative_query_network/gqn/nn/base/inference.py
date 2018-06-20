class Network:
    def forward_onestep(self, prev_h, prev_c, prev_u, prev_z, v, r):
        raise NotImplementedError

    def sample_z(self, h):
        raise NotImplementedError