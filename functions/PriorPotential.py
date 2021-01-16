import numpy as np
from functions.Function import Function


class PriorPotential(Function):
    def __init__(self, f, prior, learn_prior=True):
        self.f = f
        self.prior = prior
        self.learn_prior = learn_prior
        if f.dimension != prior.dimension:
            raise Exception('dimension does not match')
        else:
            self.dimension = f.dimension

    def __call__(self, *x):
        return self.f(*x) * self.prior(*x)

    def batch_call(self, x):
        return self.f.batch_call(x) * self.prior.batch_call(x)

    def parameters(self):
        return (self.f.parameters(), self.prior.parameters())

    def set_parameters(self, parameters):
        self.f.set_parameters(*parameters[0])
        self.prior.set_parameters(*parameters[1])
