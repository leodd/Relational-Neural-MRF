from functions.NeuralNet import *
from functions.Potentials import GaussianFunction, TableFunction, CategoricalGaussianFunction, CNNPotential, FCPotential
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

    def set_parameters(self, parameters):
        self.nn.set_parameters(parameters)

    def parameters(self):
        return self.nn.parameters()

    def update(self, dy, optimizer):
        self.nn.update(dy.reshape(-1, 1), optimizer)


class ExpWrapper(Function):
    """
    A wrapper for log function, such that the function call will return the value of exp(f(x)).
    """
    def __init__(self, f, dimension=None, formula=None):
        self.f = f
        self.dimension = dimension if dimension else f.dimension
        self.formula = formula

    def __call__(self, *x):
        return self.batch_call(np.expand_dims(np.array(x), axis=0)).squeeze()

    def batch_call(self, x):
        if self.formula: x = self.formula(x)
        return np.exp(self.f.batch_call(x))

    def log_batch_call(self, x):
        if self.formula: x = self.formula(x)
        return self.f.batch_call(x)

    def set_parameters(self, parameters):
        self.f.set_parameters(parameters)

    def parameters(self):
        return self.f.parameters()

    def update(self, dy, optimizer):
        self.f.update(dy, optimizer)


class FuncWrapper(Function):
    """
    A wrapper for function, enabling feature mapping.
    """
    def __init__(self, f, dimension=None, formula=None):
        self.f = f
        self.dimension = dimension if dimension else f.dimension
        self.formula = formula

    def __call__(self, *x):
        return self.batch_call(np.expand_dims(np.array(x), axis=0)).squeeze()

    def batch_call(self, x):
        if self.formula: x = self.formula(x)
        return self.f.batch_call(x)

    def log_batch_call(self, x):
        if self.formula: x = self.formula(x)
        return self.f.log_batch_call(x)

    def set_parameters(self, parameters):
        self.f.set_parameters(parameters)

    def parameters(self):
        return self.f.parameters()

    def update(self, dy, optimizer):
        self.f.update(dy, optimizer)
