from Function import Function
from Potentials import GaussianFunction
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

    def model_parameters(self):
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
        return np.exp(self.nn.forward(x, save_cache=False))

    def set_parameters(self, parameters):
        self.nn.set_parameters(parameters)

    def model_parameters(self):
        return self.nn.model_parameters()


class GaussianNeuralNetPotential(Function):
    """
    A wrapper for NeuralNetFunction class, such that the function call will return the value of exp(nn(x)).
    """
    def __init__(self, *args):
        self.dimension = args[0][0]  # The dimension of the input parameters
        self.nn = NeuralNetFunction(*args)
        self.gaussian = GaussianFunction(np.ones(self.dimension), np.eye(self.dimension))

    def __call__(self, *parameters):
        return np.exp(self.nn(*parameters)) * self.gaussian(*parameters)

    def batch_call(self, x):
        return np.exp(self.nn.forward(x, save_cache=False)) * self.gaussian.batch_call(x)

    def set_parameters(self, parameters):
        self.nn.set_parameters(parameters[0])
        self.gaussian.set_parameters(*parameters[1])

    def model_parameters(self):
        return (
            self.nn.model_parameters(),
            (self.gaussian.mu, self.gaussian.sig)
        )
