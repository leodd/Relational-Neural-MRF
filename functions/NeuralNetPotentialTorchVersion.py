from functions.Function import Function
import numpy as np
import torch
import torch.nn as nn
from functions.Potentials import GaussianFunction


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

        self.out = None  # The cache for the batch call output

    def parameters(self):
        return self.nn.state_dict()

    def set_parameters(self, parameters):
        self.nn.load_state_dict(parameters)

    def __call__(self, *parameters):
        x = torch.Tensor(parameters).reshape(1, -1).to(self.device)
        return torch.exp(self.nn(x)).reshape(-1)

    def batch_call(self, x):
        return torch.exp(self.nn(x)).reshape(-1)


class GaussianNeuralNetPotential(Function):
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

        self.gaussian = GaussianFunction(np.ones(self.dimension), np.eye(self.dimension))

    def parameters(self):
        return (
            self.nn.state_dict(),
            (self.gaussian.mu, self.gaussian.sig)
        )

    def set_parameters(self, parameters):
        self.nn.load_state_dict(parameters[0])
        self.gaussian.set_parameters(*parameters[1])

    def __call__(self, *parameters):
        x = torch.FloatTensor(parameters).reshape(1, -1).to(self.device)
        return torch.exp(self.nn(x)).reshape(-1) * torch.from_numpy(self.gaussian(*parameters)).float().to(self.device)

    def batch_call(self, x):
        return torch.exp(self.nn(x)).reshape(-1) * \
               torch.from_numpy(self.gaussian.batch_call(x.cpu().numpy())).float().to(self.device)
