from functions.NeuralNetPotential import NeuralNetPotential, ReLU, LinearLayer
from optimization_tools import AdamOptimizer
from utils import visualize_1d_potential
from Graph import *
import numpy as np
import seaborn as sns

domain = Domain([-20, 10], continuous=True)

def prior(x):
    return np.ones(x.shape) / (domain.values[1] - domain.values[0])

data = np.concatenate([
    np.random.normal(loc=0, scale=3, size=300),
    np.random.normal(loc=-10, scale=3, size=300)
])

# data = np.random.normal(loc=-5, scale=3, size=1000)

print(len(data))

data = np.array(data)

sns.distplot(data)

p = NeuralNetPotential(
    layers=[LinearLayer(1, 64), ReLU(),
            LinearLayer(64, 32), ReLU(),
            LinearLayer(32, 1)],
)

num_samples = 10
alpha = 0.5

def create_training_data(log=False):
    nn_data = np.empty([len(data) + num_samples, 1])

    nn_data[num_samples:, 0] = data

    shift = np.random.rand()
    w = (domain.values[1] - domain.values[0]) / (num_samples - 1)
    points = np.linspace(domain.values[0], domain.values[1], num=num_samples, endpoint=True) + (shift - 0.5) * w
    points[0], points[-1] = domain.values[0] + shift * w * 0.5, domain.values[1] - (1 - shift) * w * 0.5
    ws = np.ones(num_samples) * w
    ws[0], ws[-1] = shift * w, (1 - shift) * w

    nn_data[:num_samples, 0] = points
    xs = points.reshape(-1, 1)

    ys = p.forward(xs, save_cache=False)
    ys = ys.reshape(-1)
    ys_ = np.exp(ys)
    ys /= np.sum(ys_ * ws)

    if log:
        visualize_1d_potential(p, domain, 0.3, True)
        # plt.plot(xs.reshape(-1), ys)
        # plt.show()

    nn_data_gradient = np.ones([len(data) + num_samples, 1]) * alpha

    # predict = nn.forward(data.reshape(-1, 1), save_cache=False).reshape(-1)
    # nn_data_gradient[num_samples:, 0] -= (1 - alpha) * (np.exp(predict) - prior(data)) * predict

    nn_data_gradient /= len(data)

    nn_data_gradient[:num_samples, 0] = -alpha * ys - (1 - alpha) * (ys_ - prior(points)) * ys_
    # nn_data_gradient[:num_samples, 0] = -alpha * ys

    return nn_data, nn_data_gradient


visualize_1d_potential(p, domain, 0.3, True)
visualize_1d_potential(p, domain, 0.3, False)

lr = 0.001
regular = 0.001

adam = AdamOptimizer(lr)
moments = dict()

for t in range(1, 10000):
    x, d_y = create_training_data(t % 500 == 0)
    p.forward(x)
    _, d_network = p.backward(d_y)

    for layer, (d_W, d_b) in d_network.items():
        step, moment = adam(d_W - layer.W * regular, moments.get((layer, 'W'), (0, 0)), t)
        layer.W += step
        moments[(layer, 'W')] = moment

        step, moment = adam(d_b - layer.b * regular, moments.get((layer, 'b'), (0, 0)), t)
        layer.b += step
        moments[(layer, 'b')] = moment
        #
        # layer.W += d_W * lr - layer.W * regular
        # layer.b += d_b * lr - layer.b * regular

    print(t)

visualize_1d_potential(p, domain, 0.3, True)
visualize_1d_potential(p, domain, 0.3, False)
