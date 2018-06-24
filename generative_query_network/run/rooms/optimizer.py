import math
from chainer import optimizers


class Optimizer:
    def __init__(
            self,
            model_parameters,
            # Learning rate at training step s with annealing
            mu_i=5.0 * 1e-4,
            mu_f=5.0 * 1e-5,
            n=1.6 * 1e6,
            # Learning rate as used by the Adam algorithm
            beta_1=0.9,
            beta_2=0.99,
            # Adam regularisation parameter
            eps=1e-8):
        self.mu_i = mu_i
        self.mu_f = mu_f
        self.n = n
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps

        lr = self.mu_s(0)
        self.optimizer = optimizers.Adam(
            lr, beta1=beta_1, beta2=beta_2, eps=eps)
        self.optimizer.setup(model_parameters)

    def mu_s(self, training_step):
        return max(
            self.mu_f +
            (self.mu_i - self.mu_f) * (1.0 - training_step / self.n),
            self.mu_f)

    def anneal_learning_rate(self, training_step):
        self.optimizer.hyperparam.alpha = self.mu_s(training_step)

    def step(self, training_step):
        self.optimizer.update()
        self.optimizer.alpha = self.mu_s(training_step)
