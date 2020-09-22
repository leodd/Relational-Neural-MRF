from NeuralNetPotential import GaussianNeuralNetPotential, TableNeuralNetPotential, LeakyReLU, ReLU, ELU, LinearLayer
from Potentials import TableFunction
from learning.NeuralPMLEHybrid import PMLE
from utils import visualize_1d_potential, visualize_2d_potential
from Graph import *
import numpy as np
import seaborn as sns


domain = Domain([0, 1], continuous=False)

rv1 = RV(domain)
rv2 = RV(domain)

choice = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])
temp = np.random.choice([0, 1, 2, 3], p=[0.1, 0.3, 0.2, 0.4], size=1000)

data = {
    rv1: choice[temp, 0],
    rv2: choice[temp, 1]
}

p = TableNeuralNetPotential(
    layers=[LinearLayer(2, 64), ReLU(),
            LinearLayer(64, 32), ReLU(),
            LinearLayer(32, 1)],
    domains=[domain, domain],
    prior=TableFunction(
        np.ones([domain.size, domain.size]) / (domain.size * domain.size)
    )
)

f = F(p, nb=[rv1, rv2])

g = Graph({rv1, rv2}, {f})

def visualize(ps, t):
    if t % 200 == 0:
        visualize_2d_potential(p, domain, domain)

leaner = PMLE(g, [p], data)
leaner.train(
    lr=0.0001,
    alpha=0.999,
    regular=0.0,
    max_iter=5000,
    batch_iter=10,
    batch_size=300,
    rvs_selection_size=1,
    sample_size=20,
    visualize=visualize
)
