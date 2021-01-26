from functions.NeuralNet import *
from functions.Potentials import GaussianFunction, TableFunction, CategoricalGaussianFunction
from functions.PriorPotential import PriorPotential
import numpy as np


class NeuralNetPotential(Function):
    """
    A wrapper for NeuralNetFunction class, such that the function call will return the value of exp(nn(x)).
    """
    def __init__(self, layers, dimension=None, formula=None):
        self.dimension = dimension if dimension else layers[0].i_size  # The dimension of the input parameters
        self.nn = NeuralNetFunction(layers)
        self.formula = formula

    def __call__(self, *x):
        return self.batch_call(np.expand_dims(np.array(x), axis=0)).squeeze()

    def batch_call(self, x):
        return np.exp(self.log_batch_call(x))

    def log_batch_call(self, x):
        if self.formula: x = self.formula(x)
        return self.nn.forward(x).reshape(-1)

    def log_backward(self, dy):
        return self.nn.backward(dy)

    def set_parameters(self, parameters):
        self.nn.set_parameters(parameters)

    def parameters(self):
        return self.nn.parameters()

    def params_gradients(self, dy):
        return self.nn.params_gradients(dy)

    def update(self, steps):
        self.nn.update(steps)


class ExpWrapper(Function):
    """
    A wrapper for normal function class, such that the function call will return the value of exp(f(x)).
    """
    def __init__(self, f):
        self.f = f
        self.dimension = f.dimension

    def __call__(self, *x):
        return self.batch_call(np.expand_dims(np.array(x), axis=0)).squeeze()

    def batch_call(self, x):
        return np.exp(self.f.batch_call(x))

    def log_batch_call(self, x):
        return self.f.batch_call(x)

    def log_backward(self, dy):
        return self.f.backward(dy)

    def set_parameters(self, parameters):
        self.f.set_parameters(parameters)

    def parameters(self):
        return self.f.parameters()

    def params_gradients(self, dy):
        return self.f.params_gradients(dy)

    def update(self, steps):
        self.f.update(steps)
