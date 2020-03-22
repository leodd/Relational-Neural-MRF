from Function import Function
import numpy as np


class NeuralNetFunction(Function):
    """
    Usage:
        nn = NeuralNetFunction(
            (in, inner, Relu),
            (inner, out, None)
        )

        for _ in range(max_iter):
            predict = nn.forward(x)
            d_loss = compute_loss_gradient(predict, target)
            _, d_network = backward(d_loss)

            for layer, (d_W, d_b) in d_network.items():
                layer.W -= d_W * lr
                layer.b -= d_b * lr
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

    def __call__(self, *parameters):
        x = np.array(parameters, dtype=float)
        x = x[np.newaxis]

        for layer in self.layers:
            x = layer.forward(x)

        return x

    def forward(self, x, save_cache=True):
        x = np.array(x, dtype=float)

        if save_cache:
            self.cache = [x]

        for layer in self.layers:
            x = layer.forward(x)
            if save_cache:
                self.cache.append(x)

        return x

    def backward(self, d_y, x=None):
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


class Relu:
    @staticmethod
    def forward(x):
        return np.maximum(0, x)

    @staticmethod
    def backward(d_y, x):
        d_x = np.array(d_y, copy=True)
        d_x[x <= 0] = 0

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
        m = d_y.shape[0]
        d_W = x.T @ d_y / m
        d_b = np.sum(d_y, axis=0)
        d_x = d_y @ self.W.T

        return d_x, (d_W, d_b)
