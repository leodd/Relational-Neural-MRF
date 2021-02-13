from functions.NeuralNet import *
from functions.Potentials import GaussianFunction, TableFunction, CategoricalGaussianFunction
from functions.PriorPotential import PriorPotential
import numpy as np


class ConditionalNeuralPotential(Function):
    """
    The conditional part will be processed by a neural net.
    """
    def __init__(self, layers, crf_domains, conditional_dimension, conditional_formula=None):
        self.crf_dim = len(crf_domains)
        self.cond_dim = conditional_dimension
        self.dimension = self.crf_dim + self.cond_dim  # The dimension of the input parameters
        self.nn = NeuralNetFunction(layers)
        self.formula = conditional_formula
        shape = [len(d.values) for d in crf_domains]
        self.out_dim = np.product(shape)
        self.table = np.arange(self.out_dim).reshape(shape)

    def __call__(self, *x):
        return self.batch_call(np.expand_dims(np.array(x), axis=0)).squeeze()

    def batch_call(self, x):
        return np.exp(self.log_batch_call(x))

    def log_batch_call(self, x):
        cond_x, crf_x = x[:, :self.cond_dim], x[:, self.cond_dim:].astype(int)
        if self.formula: cond_x = self.formula(cond_x)
        y = self.nn.forward(cond_x)
        idx = self.table[tuple(crf_x.T)]
        if setting.save_cache:
            self.cache = idx
        return y[np.arange(len(y)), idx]

    def set_parameters(self, parameters):
        self.nn.set_parameters(parameters)

    def parameters(self):
        return self.nn.parameters()

    def params_gradients(self, dy):
        dy_nn = np.zeros([len(dy), self.out_dim])
        dy_nn[np.arange(len(dy)), self.cache] = dy
        return self.nn.params_gradients(dy_nn)

    def update(self, steps):
        self.nn.update(steps)
