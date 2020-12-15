from Function import Function
from NeuralNet import *
from Potentials import GaussianFunction, TableFunction, CategoricalGaussianFunction, LinearGaussianFunction
import numpy as np
from numpy.linalg import det, inv


class NeuralNetPotential(Function):
    """
    A wrapper for NeuralNetFunction class, such that the function call will return the value of exp(nn(x)).
    """
    def __init__(self, layers):
        self.dimension = layers[0].i_size  # The dimension of the input parameters
        self.nn = NeuralNetFunction(layers)

    def __call__(self, *x):
        return np.exp(self.nn(*x))

    def batch_call(self, x):
        return np.exp(self.nn.forward(x, save_cache=False)).reshape(-1)

    def nn_forward(self, x, save_cache=True):
        return self.nn.forward(x, save_cache)

    def nn_backward(self, dy):
        return self.nn.backward(dy)

    def set_parameters(self, parameters):
        self.nn.set_parameters(parameters)

    def parameters(self):
        return self.nn.parameters()


class GaussianNeuralNetPotential(Function):
    def __init__(self, layers, prior=None, eps=0):
        self.dimension = layers[0].i_size  # The dimension of the input parameters
        self.nn = NeuralNetFunction(layers)
        self.prior = prior
        self.eps = eps

    def __call__(self, *x):
        return np.exp(self.nn(*x)) * self.prior(*x) + self.eps

    def batch_call(self, x):
        return np.exp(self.nn.forward(x, save_cache=False)).reshape(-1) * self.prior.batch_call(x) + self.eps

    def nn_forward(self, x, save_cache=True):
        return self.nn.forward(x, save_cache)

    def nn_backward(self, dy):
        return self.nn.backward(dy)

    def prior_slice(self, *parameters):
        return self.prior.slice(*parameters)

    def set_empirical_prior(self, data):
        mu = np.mean(data, axis=0).reshape(-1)
        sig = np.cov(data.T).reshape(self.dimension, self.dimension)

        if self.prior is None:
            self.prior = GaussianFunction(mu, sig, eps=0)
        else:
            self.prior.set_parameters(mu, sig)

    def set_parameters(self, parameters):
        self.nn.set_parameters(parameters[0])

        if self.prior is None:
            self.prior = GaussianFunction(*parameters[1], eps=0)
        else:
            self.prior.set_parameters(*parameters[1])

    def parameters(self):
        return (
            self.nn.parameters(),
            (self.prior.mu, self.prior.sig)
        )


class TableNeuralNetPotential(Function):
    def __init__(self, layers, domains, prior=None, eps=0):
        self.dimension = layers[0].i_size  # The dimension of the input parameters
        self.nn = NeuralNetFunction(layers)
        self.domains = domains
        self.prior = prior
        self.eps = eps

    def __call__(self, *x):
        return np.exp(self.nn(*x)) * self.prior(*x) + self.eps

    def batch_call(self, x):
        return np.exp(self.nn.forward(x, save_cache=False)).reshape(-1) * self.prior.batch_call(x) + self.eps

    def nn_forward(self, x, save_cache=True):
        return self.nn.forward(x, save_cache)

    def nn_backward(self, dy):
        return self.nn.backward(dy)

    def prior_slice(self, *parameters):
        return self.prior.slice(*parameters)

    def set_empirical_prior(self, data):
        data = data.astype(int)
        table = np.zeros(shape=[len(d.values) for d in self.domains])

        idx, count = np.unique(data, return_counts=True, axis=0)
        table[tuple(idx.T)] = count
        table /= np.sum(table)

        if self.prior is None:
            self.prior = TableFunction(table)
        else:
            self.prior.table = table

    def set_parameters(self, parameters):
        self.nn.set_parameters(parameters[0])

        if self.prior is None:
            self.prior = TableFunction(parameters[1])
        else:
            self.prior.table = parameters[1]

    def parameters(self):
        return (
            self.nn.parameters(),
            self.prior.table
        )


class CGNeuralNetPotential(Function):
    def __init__(self, layers, domains, prior=None, eps=0):
        self.dimension = layers[0].i_size  # The dimension of the input parameters
        self.nn = NeuralNetFunction(layers)
        self.domains = domains
        self.prior = prior
        self.eps = eps

    def __call__(self, *x):
        return np.exp(self.nn(*x)) * self.prior(*x) + self.eps

    def batch_call(self, x):
        return np.exp(self.nn.forward(x, save_cache=False)).reshape(-1) * self.prior.batch_call(x) + self.eps

    def nn_forward(self, x, save_cache=True):
        return self.nn.forward(x, save_cache)

    def nn_backward(self, dy):
        return self.nn.backward(dy)

    def prior_slice(self, *parameters):
        return self.prior.slice(*parameters)

    def set_empirical_prior(self, data):
        c_idx = [i for i, d in enumerate(self.domains) if d.continuous]
        d_idx = [i for i, d in enumerate(self.domains) if not d.continuous]

        w_table = np.zeros(shape=[len(self.domains[i].values) for i in d_idx])
        dis_table = np.zeros(shape=w_table.shape, dtype=int)

        idx, count = np.unique(data[:, d_idx].astype(int), return_counts=True, axis=0)
        w_table[tuple(idx.T)] = count
        w_table /= np.sum(w_table)

        dis = [GaussianFunction(np.zeros(len(d_idx)), np.eye(len(d_idx)))]

        for row in idx:
            row_idx = np.where(np.all(data[:, d_idx] == row, axis=1))
            row_data = data[row_idx][:, c_idx]

            if len(row_data) <= 1:
                continue

            mu = np.mean(row_data, axis=0).reshape(-1)
            sig = np.cov(row_data.T).reshape(len(c_idx), len(c_idx))

            dis_table[tuple(row)] = len(dis)
            dis.append(GaussianFunction(mu, sig + 2))

        if self.prior is None:
            self.prior = CategoricalGaussianFunction(w_table, dis_table, dis, self.domains)
        else:
            self.prior.set_parameters(w_table, dis_table, dis, self.domains)

    def set_parameters(self, parameters):
        self.nn.set_parameters(parameters[0])

        if self.prior is None:
            self.prior = CategoricalGaussianFunction(*parameters[1], self.domains)
        else:
            self.prior.set_parameters(*parameters[1], self.domains)

    def parameters(self):
        return (
            self.nn.parameters(),
            (self.prior.w_table, self.prior.dis_table, self.prior.dis)
        )


class ContrastiveNeuralNetPotential(Function):
    def __init__(self, layers, prior=None, eps=0):
        self.dimension = 2  # The dimension of the input parameters
        self.nn = NeuralNetFunction(layers)
        self.prior = prior
        self.eps = eps

    def __call__(self, *parameters):
        x = np.abs(parameters[0] - parameters[1])
        return np.exp(self.nn(x)) * self.prior(x) + self.eps

    def batch_call(self, x):
        x = np.abs(x[:, [0]] - x[:, [1]])
        return np.exp(self.nn.forward(x, save_cache=False)).reshape(-1) * self.prior.batch_call(x) + self.eps

    def nn_forward(self, x, save_cache=True):
        x = np.abs(x[:, [0]] - x[:, [1]])
        return self.nn.forward(x, save_cache)

    def nn_backward(self, dy):
        return self.nn.backward(dy)

    def prior_slice(self, *parameters):
        if parameters[0] is not None:
            return GaussianFunction([parameters[0]], self.prior.sig, eps=0)
        else:
            return GaussianFunction([parameters[1]], self.prior.sig, eps=0)

    def set_empirical_prior(self, data):
        data = data[:, 0] - data[:, 1]
        sig = np.var(data).reshape(1, 1)

        if self.prior is None:
            self.prior = GaussianFunction([0], sig, eps=0)
        else:
            self.prior.set_parameters([0], sig)

    def set_parameters(self, parameters):
        self.nn.set_parameters(parameters[0])

        if self.prior is None:
            self.prior = GaussianFunction([0], parameters[1], eps=0)
        else:
            self.prior.set_parameters([0], parameters[1])

    def parameters(self):
        return (
            self.nn.parameters(),
            self.prior.sig
        )
