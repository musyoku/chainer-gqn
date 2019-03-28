import math

import chainer
import chainermn
from chainer import optimizers
from chainer.optimizer_hooks import GradientClipping
from tabulate import tabulate


class AdamOptimizer:
    def __init__(
            self,
            model_parameters,
            # Learning rate at training step s with annealing
            initial_lr=1e-4,
            final_lr=1e-5,
            annealing_steps=1600000,
            # Learning rate as used by the Adam algorithm
            beta_1=0.9,
            beta_2=0.99,
            # Adam regularisation parameter
            eps=1e-8,
            initial_training_step=0,
            communicator=None):
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.annealing_steps = annealing_steps
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps

        lr = self.compute_lr_at_step(initial_training_step)
        self.optimizer = optimizers.Adam(
            lr, beta1=beta_1, beta2=beta_2, eps=eps)
        self.optimizer.setup(model_parameters)

        self.multi_node_optimizer = None
        if communicator:
            self.multi_node_optimizer = chainermn.create_multi_node_optimizer(
                self.optimizer, communicator)

    @property
    def learning_rate(self):
        return self.optimizer.alpha

    def compute_lr_at_step(self, training_step):
        return max(
            self.final_lr + (self.initial_lr - self.final_lr) *
            (1.0 - training_step / self.annealing_steps), self.final_lr)

    def anneal_learning_rate(self, training_step):
        self.optimizer.hyperparam.alpha = self.compute_lr_at_step(
            training_step)

    def update(self, training_step):
        if self.multi_node_optimizer:
            self.multi_node_optimizer.update()
        else:
            self.optimizer.update()
        self.anneal_learning_rate(training_step)

    def __str__(self):
        rows = []
        for key, value in self.__dict__.items():
            rows.append([key, value])
        return tabulate(rows, headers=["Optimizer", ""])
