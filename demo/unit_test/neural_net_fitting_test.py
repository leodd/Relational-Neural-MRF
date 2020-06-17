from NeuralNetPotential import NeuralNetPotential, ReLU, ELU, LeakyReLU
import numpy as np
from Graph import *
from utils import visualize_1d_potential
from optimization_tools import AdamOptimizer


def norm_pdf(x, mu, sig):
    u = (x - mu)
    y = np.exp(-u * u * 0.5 / sig) / (2.506628274631 * sig)
    return y


m = 100
x = np.random.uniform(-15, 15, size=(m, 1))
target = norm_pdf(x, 0, 9)
print(norm_pdf(0, 0, 9))
lr = 0.01

nn = NeuralNetPotential(
    (1, 32, LeakyReLU()),
    (32, 64, LeakyReLU()),
    (64, 32, LeakyReLU()),
    (32, 1, None)
)

adam = AdamOptimizer(lr)
moments = dict()
regular = 0.001

for t in range(1, 5000):
    predict = np.exp(nn.forward(x))
    d_loss = (predict - target) * predict
    _, d_network = nn.backward(d_loss)

    for layer, (d_W, d_b) in d_network.items():
        step, moment = adam(d_W / m, moments.get((layer, 'W'), (0, 0)), t)
        layer.W -= step + layer.W * regular
        moments[(layer, 'W')] = moment

        step, moment = adam(d_b / m, moments.get((layer, 'b'), (0, 0)), t)
        layer.b -= step + layer.b * regular
        moments[(layer, 'b')] = moment

    print(np.sum((predict - target) ** 2))

visualize_1d_potential(nn, Domain([-15, 15]), 0.3, is_exp=True)
