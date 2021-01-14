import numpy as np
from functions.Function import Function


class PriorPotential(Function):
    def __init__(self, f, prior, learn_prior=False):
        self.f = f
        self.prior = prior
        self.learn_prior = learn_prior
        if f.dimension != prior.dimension:
            raise Exception('dimension does not match')
        else:
            self.dimension = f.dimension

    def __call__(self, *x):
        self.f(*x) * self.prior(*x)

    def batch_call(self, x):
        self.f.forward(x) * self.prior.batch_call(x)
