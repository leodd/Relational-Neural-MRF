from NeuralNetPotential import GaussianNeuralNetPotential, TableNeuralNetPotential, LeakyReLU, ReLU, ELU, LinearLayer
from Potentials import TableFunction
from learning.NeuralPMLEHybrid import PMLE
from utils import visualize_1d_potential
from Graph import *
import numpy as np
import seaborn as sns


domain = Domain([0, 1, 2, 3], continuous=False)

rv = RV(domain)

data = {
    rv: np.random.choice([0, 1, 2, 3], p=[0.1, 0.3, 0.2, 0.4], size=1000)
}

sns.distplot(data[rv])

p = TableNeuralNetPotential(
    layers=[LinearLayer(1, 64), ReLU(),
            LinearLayer(64, 32), ReLU(),
            LinearLayer(32, 1)],
    domains=[domain],
    prior=TableFunction(
        np.ones([domain.size]) / domain.size
    )
)

f = F(p, nb=[rv])

g = Graph({rv}, {f})

def visualize(ps, t):
    if t % 200 == 0:
        visualize_1d_potential(p, domain)

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
