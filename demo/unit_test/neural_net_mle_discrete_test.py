from functions.NeuralNetPotential import TableNeuralNetPotential, ReLU, LinearLayer, NormalizeLayer
from functions.Potentials import TableFunction
from learner.NeuralPMLEHybrid import PMLE
from Graph import *
import numpy as np
import matplotlib.pyplot as plt

d1 = Domain([0, 1], continuous=False)
d2 = Domain([0, 1, 2, 3, 4], continuous=False)

rv1 = RV(d1)
rv2 = RV(d2)

choice = np.array([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)])
temp = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], p=[0.05, 0.1, 0.2, 0.1, 0.05, 0.2, 0.1, 0.05, 0.1, 0.05], size=4000)

# choice = np.array([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)])
# temp = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.1, 0.2, 0.1, 0.3, 0.1, 0.2], size=4000)

data = {
    rv1: choice[temp, 0],
    rv2: choice[temp, 1]
}

p = TableNeuralNetPotential(
    layers=[
        NormalizeLayer([(-1, 1)] * 2, domains=[d1, d2]),
        # WSLinearLayer([[0, 1]], 64), ReLU(),
        LinearLayer(2, 64), ReLU(),
        LinearLayer(64, 32), ReLU(),
        LinearLayer(32, 1)
    ],
    domains=[d1, d2],
    prior=TableFunction(
        np.ones([d1.size, d2.size]) / (d1.size * d2.size)
    )
)

f = F(p, nb=[rv1, rv2])

g = Graph({rv1, rv2}, {f})

def visualize(ps, t):
    if t % 200 == 0:
        x1, x2 = np.meshgrid(d1.values, d2.values)
        xs = np.array([x1, x2]).T.reshape(-1, 2)
        ys = p.batch_call(xs).reshape(-1, 1)
        ys = ys / np.sum(ys)
        print(np.concatenate([xs, ys], axis=1))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs[:, 0], xs[:, 1], ys)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('value')
        plt.show()

        # visualize_2d_potential(p, d1, d2)

leaner = PMLE(g, [p], data)
leaner.train(
    lr=0.001,
    alpha=1.,
    regular=0.0,
    max_iter=5000,
    batch_iter=10,
    batch_size=300,
    rvs_selection_size=2,
    sample_size=20,
    visualize=visualize
)
