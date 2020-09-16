import numpy as np
from NeuralNetPotential import NeuralNetFunction, ReLU, ELU, LeakyReLU, LinearLayer
import scipy.integrate as integrate
from optimization_tools import AdamOptimizer


m = 3
x = np.random.randn(m, 1)
target = np.random.randn(m, 1)
print(x, target)
lr = 0.01

nn = NeuralNetFunction(
    layers=[LinearLayer(1, 64), ReLU(),
            LinearLayer(64, 32), ReLU(),
            LinearLayer(32, 1)]
)

adam = AdamOptimizer(lr)
moments = dict()

for t in range(1, 1000):
    predict = nn.forward(x)
    d_loss = predict - target
    _, d_network = nn.backward(d_loss)

    for layer, (d_W, d_b) in d_network.items():
        step, moment = adam(d_W / m, moments.get((layer, 'W'), (0, 0)), t)
        layer.W -= step
        moments[(layer, 'W')] = moment

        step, moment = adam(d_b / m, moments.get((layer, 'b'), (0, 0)), t)
        layer.b -= step
        moments[(layer, 'b')] = moment

    # print(np.sum((predict - target) ** 2))


mb = [1]
idx = 0


def riemann_integration(lower_bound, upper_bound, num_samples, shift=0.5):
    w = (upper_bound - lower_bound) / (num_samples - 1)

    points = np.linspace(lower_bound, upper_bound, num=num_samples, endpoint=True) + (shift - 0.5) * w
    points[0], points[-1] = lower_bound + shift * w * 0.5, upper_bound - (1 - shift) * w * 0.5
    ws = np.ones(num_samples) * w
    ws[0], ws[-1] = shift * w, (1 - shift) * w

    xs = np.empty([num_samples, 1])
    xs[:, :] = mb
    xs[:, idx] = points

    ys = np.exp(nn.forward(xs, save_cache=False))
    ys.shape = (num_samples,)

    return np.sum(ys * ws)


def trapezoidal_integration(lower_bound, upper_bound, num_samples, shift=0):
    points = np.linspace(lower_bound, upper_bound, num=num_samples + 2, endpoint=True)
    points[1:-1] += shift * (upper_bound - lower_bound) / (num_samples + 1)
    ws = points[1:] - points[:-1]

    xs = np.empty([num_samples + 2, 1])
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


res = riemann_integration(0, 1, 10, 0.5)
print(res)

# res = riemann_integration(0, 1, 10, 0)
# print(res)

res = trapezoidal_integration(0, 1, 10)
print(res)

# Compare with scipy integral approximation
res = integrate.quad(lambda v: np.exp(nn(v)), 0, 1)
print(res)
