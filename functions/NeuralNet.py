from functions.Function import Function
from functions.setting import train_mod
import functions.setting as setting
import numpy as np


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
    def __init__(self, layers):
        Function.__init__(self)

        self.layers = layers

    def set_parameters(self, parameters):
        idx = 0

        for layer in self.layers:
            if getattr(layer, 'set_parameters', False):
                layer.set_parameters(parameters[idx])
                idx += 1

    def parameters(self):
        parameters = list()

        for layer in self.layers:
            if getattr(layer, 'parameters', False):
                parameters.append(
                    layer.parameters()
                )

        return parameters

    def __call__(self, *parameters):
        x = np.array(parameters, dtype=float)
        x = x[np.newaxis]

        for layer in self.layers:
            x = layer.forward(x)

        return x

    def batch_call(self, x):
        return self.forward(x)

    def forward(self, x):  # x must be numpy array
        if setting.save_cache:
            self.cache = [x]

        for layer in self.layers:
            x = layer.forward(x)
            if setting.save_cache:
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

    def params_gradients(self, d_y):
        d_x = d_y
        params = list()
        gradients = list()

        for idx in reversed(range(len(self.layers))):
            layer = self.layers[idx]
            x = self.cache[idx]
            d_x, d_param = layer.backward(d_x, x)
            if d_param is not None:
                params.extend(layer.parameters())
                gradients.extend(d_param)

        return params, gradients

    def update(self, steps):
        i = 0
        for idx in reversed(range(len(self.layers))):
            layer = self.layers[idx]
            if getattr(layer, 'parameters', False):
                W, b = steps[i], steps[i + 1]
                layer.W += W
                layer.b += b
                i += 2


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

    def parameters(self):
        return self.W, self.b

    def set_parameters(self, parameters):
        self.W, self.b = parameters

class WSLinearLayer:
    def __init__(self, grouped_input_idx, o_size):
        self.i_size = len([_ for group in grouped_input_idx for _ in group])
        self.o_size = o_size
        self.W = np.random.randn(len(grouped_input_idx), o_size) * 0.1
        self.b = np.random.randn(o_size) * 0.1

        self.forward_mapper = np.zeros(self.i_size, dtype=int)
        self.backward_mapper = np.zeros([len(grouped_input_idx), self.i_size])
        for idx, group_idx in enumerate(grouped_input_idx):
            self.forward_mapper[group_idx] = idx
            self.backward_mapper[idx, group_idx] = 1 / len(group_idx)

    def forward(self, x):
        return x @ self.W[self.forward_mapper, :] + self.b

    def backward(self, d_y, x):
        d_W = x.T @ d_y
        d_W = self.backward_mapper @ d_W

        d_b = np.sum(d_y, axis=0)
        d_x = d_y @ self.W[self.forward_mapper, :].T

        return d_x, (d_W, d_b)

    def parameters(self):
        return self.W, self.b

    def set_parameters(self, parameters):
        self.W, self.b = parameters


class NormalizeLayer:
    def __init__(self, ranges, domains):
        self.i_size = len(ranges)
        self.z = np.array([
            (r[-1] - r[0]) / (d.values[-1] - d.values[0]) if r is not None else 1
            for r, d in zip(ranges, domains)
        ])
        self.s = np.array([r[0] if r is not None else 0 for r in ranges])
        self.s_ = np.array([d.values[0] if r is not None else 0 for r, d in zip(ranges, domains)])

    def forward(self, x):
        return (x - self.s_) * self.z + self.s

    def backward(self, d_y, x):
        return self.z * d_y, None


class Clamp:
    def __init__(self, a, b):
        self.a, self.b = a, b

    def forward(self, x):
        return np.clip(x, self.a, self.b)

    def backward(self, d_y, x):
        d_x = np.array(d_y, copy=True)
        d_x[(x < self.a) & (d_y < 0)] = 0
        d_x[(x > self.b) & (d_y > 0)] = 0
        return d_x, None
