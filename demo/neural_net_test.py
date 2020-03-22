from NeuralNetPotential import NeuralNetFunction, Relu
import numpy as np
from optimization_tools import AdamOptimizer


m = 100
x = np.random.randn(m, 3)
target = np.random.randn(m, 1)
lr = 0.01

nn = NeuralNetFunction(
    (3, 4, Relu),
    (4, 4, Relu),
    (4, 1, None)
)

adam = AdamOptimizer(lr)
moments = dict()

for t in range(1, 1000):
    predict = nn.forward(x)
    d_loss = predict - target
    _, d_network = nn.backward(d_loss)

    for layer, (d_W, d_b) in d_network.items():
        step, moment = adam(d_W, moments.get((layer, 'W'), (0, 0)), t)
        layer.W -= step
        moments[(layer, 'W')] = moment

        step, moment = adam(d_b, moments.get((layer, 'b'), (0, 0)), t)
        layer.b -= step
        moments[(layer, 'b')] = moment

    print(np.sum((predict - target) ** 2))
