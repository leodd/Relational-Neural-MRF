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
        return (self.nn.parameters(),)
