from functions.ExpPotentials import PriorPotential, NeuralNetPotential, ReLU, LinearLayer, NormalizeLayer, train_mod
from functions.Potentials import TableFunction
from learner.NeuralPMLE import PMLE
from Graph import *
import numpy as np
import matplotlib.pyplot as plt
from utils import visualize_2d_potential

d1 = Domain([0, 1], continuous=False)
d2 = Domain([0, 1, 2, 3, 4], continuous=False)

rv1 = RV(d1)
rv2 = RV(d2)

choice = np.array([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)])
temp = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], p=[0.05, 0.1, 0.2, 0.1, 0.05, 0.2, 0.1, 0.05, 0.1, 0.05], size=5000)

# choice = np.array([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)])
# temp = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.1, 0.2, 0.1, 0.3, 0.1, 0.2], size=4000)

data = {
    rv1: choice[temp, 0],
    rv2: choice[temp, 1]
}

# p = PriorPotential(
#     NeuralNetPotential(
#         layers=[
#             NormalizeLayer([(-1, 1)] * 2, domains=[d1, d2]),
#             # WSLinearLayer([[0, 1]], 64), ReLU(),
#             LinearLayer(2, 64), ReLU(),
#             LinearLayer(64, 32), ReLU(),
#             LinearLayer(32, 1)
#         ]
#     ),
#     TableFunction(
#         np.ones([d1.size, d2.size]) / (d1.size * d2.size)
#     ),
#     learn_prior=True
# )

p = NeuralNetPotential(
    layers=[
        NormalizeLayer([(-1, 1)] * 2, domains=[d1, d2]),
        # WSLinearLayer([[0, 1]], 64), ReLU(),
        LinearLayer(2, 64), ReLU(),
        LinearLayer(64, 32), ReLU(),
        LinearLayer(32, 1)
    ]
)

f = F(p, nb=[rv1, rv2])

g = Graph({rv1, rv2}, {f})

def visualize(ps, t):
    if t % 200 == 0:
        visualize_2d_potential(p, d1, d2)
        print(p.batch_call(choice))

train_mod(True)
leaner = PMLE(g, [p], data)
leaner.train(
    lr=0.001,
    alpha=0.99,
    regular=0.0001,
    max_iter=5000,
    batch_iter=3,
    batch_size=50,
    rvs_selection_size=2,
    sample_size=20,
    visualize=visualize
)
