from NeuralNetPotential import GaussianNeuralNetPotential, LeakyReLU, ReLU, ELU
from learning.PseudoMLEWithPrior import PseudoMLELearner
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

p = GaussianNeuralNetPotential(
    (1, 64, ReLU()),
    (64, 64, ReLU()),
    (64, 64, ReLU()),
    (64, 1, None)
)

f = F(p, nb=[rv])

g = Graph({rv}, {f})

visualize_1d_potential(p, domain, 0.3)
visualize_1d_potential(p, domain, 0.3)

leaner = PseudoMLELearner(g, {p}, data)
leaner.train(
    lr=0.0001,
    alpha=0.8,
    regular=0.0001,
    max_iter=5000,
    batch_iter=10,
    batch_size=100,
    rvs_selection_size=1,
    sample_size=100
)

visualize_1d_potential(p, domain, 0.3)
visualize_1d_potential(p, domain, 0.3)
