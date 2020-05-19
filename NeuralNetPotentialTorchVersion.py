from Function import Function
import numpy as np
import torch
import torch.nn as nn


class NeuralNetModule(nn.Module):
    def __init__(self, *args):
        super(NeuralNetModule, self).__init__()

        self.layers = []

        for i_size, o_size, act in args:
            """
            i_size: input size
            o_size: output size
            act: activation function
            """
            self.layers.append(
                nn.Linear(i_size, o_size)
            )

            if act is not None:
                self.layers.append(act)

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        out = x

        for layer in self.layers:
            out = layer(out)

        return out


class NeuralNetPotential(Function):
    """
    A wrapper for PyTorch neural net, such that the function call will return the value of exp(nn(x)).
    """
    def __init__(self, *args, device=None):
        Function.__init__(self)
        self.dimension = args[0][0]  # The dimension of the input parameters

        self.device = None if device is None else torch.device(device)
        self.nn = NeuralNetModule(args)  # The wrapped neural net

    def parameters(self):
        return self.nn.parameters()

    def __call__(self, *parameters):
        return torch.exp(self.nn(parameters))

    def forward(self, x):
        if self.device is not None:
            x = x.to(self.device)
            return self.nn(x).cpu()
        else:
            return self.nn(x)

    def backward(self, d_y):
        if self.device is not None:
            d_y = d_y.to(self.device)
            return self.nn.backward(d_y).cpu()
        else:
            return self.nn.backward(d_y)
