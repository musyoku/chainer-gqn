class Optimizer:
    def __init__(
            self,
            # Learning rate at training step s with annealing
            mu_i=5.0 * 0.1**4,
            mu_f=5.0 * 0.1**4,
            n=1.6 * 10 * 6,
            # Learning rate as used by the Adam algorithm
            beta_1=0.9,
            beta_2=0.99,
            # Adam regularisation parameter
            eps=0.1**8):
        pass