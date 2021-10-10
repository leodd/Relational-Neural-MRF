from functions.ExpPotentials import NeuralNetPotential, ReLU, LinearLayer, train_mod
from functions.PriorPotential import PriorPotential
from functions.Potentials import GaussianFunction
from functions.NeuralNet import *
from learner.NeuralPMLE import PMLE
from utils import visualize_1d_potential
from Graph import *
import numpy as np
import seaborn as sns


domain = Domain([-20, 20], continuous=True)

rv = RV(domain)

data = {
    rv: np.concatenate([
        np.random.normal(loc=7, scale=3, size=300),
        np.random.normal(loc=-7, scale=3, size=300)
    ])
}

sns.distplot(data[rv])

p = PriorPotential(
    NeuralNetPotential(
        layers=[LinearLayer(1, 64), ReLU(),
                LinearLayer(64, 32), ReLU(),
                LinearLayer(32, 1), Clamp(-3, 3)]
    ),
    GaussianFunction(np.ones(1), np.ones([1, 1]))
)

# p = NeuralNetPotential(
#     layers=[LinearLayer(1, 64), ReLU(),
#             LinearLayer(64, 32), ReLU(),
#             LinearLayer(32, 1), Clamp(-3, 3)]
# )

f = F(p, nb=[rv])

g = Graph({rv}, {f})

def visualize(ps, t):
    if t % 200 == 0:
        visualize_1d_potential(p, domain, 0.3)

train_mod(True)
leaner = PMLE(g, [p], data)
leaner.train(
    lr=0.0001,
    alpha=0.999,
    regular=0.0001,
    max_iter=5000,
    batch_iter=10,
    batch_size=300,
    rvs_selection_size=1,
    sample_size=30,
    visualize=visualize
)
