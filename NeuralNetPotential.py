from Function import Function
from Potentials import GaussianFunction, TableFunction, CategoricalGaussianFunction
import numpy as np
from numpy.linalg import det, inv


class NeuralNetFunction(Function):
    """
    Usage:
        nn = NeuralNetFunction(
            (in, inner, RELU),
            (inner, out, None)
        )

        for _ in range(max_iter):
            predict = nn.forward(x)
            d_loss = compute_loss_gradient(predict, target)
            _, d_network = backward(d_loss)

            for layer, (d_W, d_b) in d_network.items():
                layer.W -= d_W * lr
                layer.b -= d_b * lr

    Note:
        The input data x is 2 dimensional,
        where the first dimension represents data point,
        and the second dimension represents features.
    """
    def __init__(self, *args):
        Function.__init__(self)

        self.layers = []

        for i_size, o_size, act in args:
            """
            i_size: input size
            o_size: output size
            act: activation function
            """
            self.layers.append(
                LinearLayer(i_size, o_size)
            )

            if act is not None:
                self.layers.append(act)

        self.cache = None  # Cache for storing the forward propagation results

    def set_parameters(self, parameters):
        idx = 0

        for layer in self.layers:
            if type(layer) is LinearLayer:
                layer.W, layer.b = parameters[idx]
                idx += 1

    def parameters(self):
        parameters = list()

        for layer in self.layers:
            if type(layer) is LinearLayer:
                parameters.append(
                    (layer.W, layer.b)
                )

        return parameters

    def __call__(self, *parameters):
        x = np.array(parameters, dtype=float)
        x = x[np.newaxis]

        for layer in self.layers:
            x = layer.forward(x)

        return x

    def forward(self, x, save_cache=True):  # x must be numpy array
        if save_cache:
            self.cache = [x]

        for layer in self.layers:
            x = layer.forward(x)
            if save_cache:
                self.cache.append(x)

        return x

    def backward(self, d_y, x=None):  # d_y must be numpy array
        if x is not None:
            self.forward(x)

        d_x = d_y
        d_network = dict()

        for idx in reversed(range(len(self.layers))):
            layer = self.layers[idx]
            x = self.cache[idx]
            d_x, d_param = layer.backward(d_x, x)
            if d_param is not None:
                d_network[layer] = d_param

        return d_x, d_network


class ReLU:
    @staticmethod
    def forward(x):
        return np.maximum(0, x)

    @staticmethod
    def backward(d_y, x):
        d_x = np.array(d_y, copy=True)
        d_x[x <= 0] = 0

        return d_x, None


class LeakyReLU:
    def __init__(self, slope=0.01):
        self.slope = slope

    def forward(self, x):
        return np.maximum(0, x) + np.minimum(0, x) * self.slope

    def backward(self, d_y, x):
        d_x = np.array(d_y, copy=True)
        d_x[x <= 0] *= self.slope

        return d_x, None


class ELU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        return np.maximum(0, x) + np.minimum(0, self.alpha * (np.exp(x) - 1))

    def backward(self, d_y, x):
        d_x = np.array(d_y, copy=True)
        temp = self.alpha * np.exp(x)
        idx = (temp - self.alpha) <= 0
        d_x[idx] *= temp[idx]

        return d_x, None


class LinearLayer:
    def __init__(self, i_size, o_size):
        self.i_size = i_size
        self.o_size = o_size
        self.W = np.random.randn(i_size, o_size) * 0.1
        self.b = np.random.randn(o_size) * 0.1

    def forward(self, x):
        return x @ self.W + self.b

    def backward(self, d_y, x):
        d_W = x.T @ d_y

        d_b = np.sum(d_y, axis=0)
        d_x = d_y @ self.W.T

        return d_x, (d_W, d_b)


class NeuralNetPotential(Function):
    """
    A wrapper for NeuralNetFunction class, such that the function call will return the value of exp(nn(x)).
    """
    def __init__(self, *args):
        self.dimension = args[0][0]  # The dimension of the input parameters
        self.nn = NeuralNetFunction(*args)

    def __call__(self, *parameters):
        return np.exp(self.nn(*parameters))

    def batch_call(self, x):
        return np.exp(self.nn.forward(x, save_cache=False)).reshape(-1)

    def set_parameters(self, parameters):
        self.nn.set_parameters(parameters)

    def parameters(self):
        return self.nn.parameters()


class GaussianNeuralNetPotential(Function):
    def __init__(self, *args):
        self.dimension = args[0][0]  # The dimension of the input parameters
        self.nn = NeuralNetFunction(*args)
        self.prior = None

    def __call__(self, *parameters):
        return np.exp(self.nn(*parameters)) * (self.prior(*parameters) + 0.001)

    def batch_call(self, x):
        return np.exp(self.nn.forward(x, save_cache=False)).reshape(-1) * (self.prior.batch_call(x) + 0.001)

    def set_empirical_prior(self, data):
        mu = np.mean(data, axis=0).reshape(-1)
        sig = np.cov(data.T).reshape(self.dimension, self.dimension)

        if self.prior is None:
            self.prior = GaussianFunction(mu, sig)
        else:
            self.prior.set_parameters(mu, sig)

    def set_parameters(self, parameters):
        self.nn.set_parameters(parameters[0])

        if self.prior is None:
            self.prior = GaussianFunction(*parameters[1])
        else:
            self.prior.set_parameters(*parameters[1])

    def parameters(self):
        return (
            self.nn.parameters(),
            (self.prior.mu, self.prior.sig)
        )


class TableNeuralNetPotential(Function):
    def __init__(self, *args, domains):
        self.dimension = args[0][0]  # The dimension of the input parameters
        self.nn = NeuralNetFunction(*args)
        self.domains = domains
        self.prior = None

    def __call__(self, *parameters):
        return np.exp(self.nn(*parameters)) * (self.prior(*parameters) + 0.001)

    def batch_call(self, x):
        return np.exp(self.nn.forward(x, save_cache=False)).reshape(-1) * (self.prior.batch_call(x) + 0.001)

    def set_empirical_prior(self, data):
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
    def __init__(self, *args, domains):
        self.dimension = args[0][0]  # The dimension of the input parameters
        self.nn = NeuralNetFunction(*args)
        self.domains = domains
        self.prior = None

    def __call__(self, *parameters):
        return np.exp(self.nn(*parameters)) * (self.prior(*parameters) + 0.001)

    def batch_call(self, x):
        return np.exp(self.nn.forward(x, save_cache=False)).reshape(-1) * (self.prior.batch_call(x) + 0.001)

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
            dis.append(GaussianFunction(mu, sig))

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
