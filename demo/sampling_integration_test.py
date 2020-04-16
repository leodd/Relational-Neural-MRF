import numpy as np
from numpy.random import uniform
from NeuralNetPotential import NeuralNetFunction, ReLU, ELU, LeakyReLU
import scipy.integrate as integrate


nn = NeuralNetFunction(
    (3, 4, LeakyReLU()),
    (4, 1, None)
)

mb = [1, -2, -3]
idx = 0


def box_integration(lower_bound, upper_bound, num_samples, shift=0.5):
    points = np.linspace(lower_bound, upper_bound, num=num_samples, endpoint=False) + shift
    w = (upper_bound - lower_bound) / num_samples

    xs = np.empty([num_samples, 3])
    xs[:, :] = mb
    xs[:, idx] = points

    return np.sum(np.exp(nn.forward(xs, save_cache=False)) * w)


def trapezoidal_integration(lower_bound, upper_bound, num_samples, shift=0.5):
    points = np.linspace(lower_bound, upper_bound, num=num_samples + 2, endpoint=True)
    points[1:-1] += shift
    ws = points[1:] - points[:-1]

    xs = np.empty([num_samples + 2, 3])
    xs[:, :] = mb
    xs[:, idx] = points

    ys = np.exp(nn.forward(xs, save_cache=False))
    ys.shape = (num_samples + 2,)
    ys = ys[1:] + ys[:-1]

    return np.sum(ys * ws) * 0.5

    # points = np.linspace(lower_bound, upper_bound, num=num_samples, endpoint=False) + shift
    # w = (upper_bound - lower_bound) / num_samples
    #
    # xs = np.empty([num_samples, 3])
    # xs[:, :] = mb
    # xs[:, idx] = points
    #
    # ys = np.exp(nn.forward(xs, save_cache=False))
    # ys.shape = (num_samples,)
    #
    # return np.sum((ys[1:] + ys[:-1])) * w * 0.5 + shift * ys[0] + (upper_bound - points[-1]) * ys[-1]


res = box_integration(-1, 1, 10)
print(res)

res = trapezoidal_integration(-1, 1, 10)
print(res)

# Compare with scipy integral approximation
res = integrate.quad(lambda v: np.exp(nn(v, -2, -3)), -1, 1)
print(res)
