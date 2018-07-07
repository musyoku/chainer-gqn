class CoreNetwork:
    def forward_onestep(self, prev_h, prev_c, prev_u, prev_z, v, r):
        raise NotImplementedError


class PriorNetwork:
    def compute_mean_z(self, z):
        raise NotImplementedError

    def compute_ln_var_z(self, h):
        raise NotImplementedError


class ObservationNetwork:
    def compute_mean_x(self, u):
        raise NotImplementedError

    def sample_x(self, u):
        raise NotImplementedError