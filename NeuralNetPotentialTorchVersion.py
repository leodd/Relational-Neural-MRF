from Function import Function
import numpy as np
import torch
import torch.nn as nn


class NeuralNetModule(nn.Module):
    def __init__(self, *args):
        super(NeuralNetModule, self).__init__()

        layers = []

        for idx, (i_size, o_size, act) in enumerate(args):
            """
            i_size: input size
            o_size: output size
            act: activation function
            """
            layers.append(
                nn.Linear(i_size, o_size)
            )

            if idx < len(args) - 1:
                layers.append(
                    nn.BatchNorm1d(o_size)
                )

            if act is not None:
                layers.append(act)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class NeuralNetPotential(Function):
    """
    A wrapper for PyTorch neural net, such that the function call will return the value of exp(nn(x)).
    """
    def __init__(self, *args, device=None):
        Function.__init__(self)
        self.dimension = args[0][0]  # The dimension of the input parameters

        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device

        self.nn = NeuralNetModule(*args)  # The wrapped neural net
        self.nn.to(device)  # Place the network on specified device

        self.out = None  # The cache out the forward output tensor

    def parameters(self):
        return self.nn.parameters()

    def __call__(self, *parameters):
        return torch.exp(self.nn(parameters))

    def forward(self, x):
        self.out = self.nn(x)
        return self.out

    def backward(self, d_y):
        return self.out.backward(d_y)
