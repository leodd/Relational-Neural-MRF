from NeuralNetPotential import NeuralNetPotential, LeakyReLU
from learning.PseudoMLE import PseudoMLELearner
from utils import visualize_1d_neural_net
from Graph import *
import numpy as np


domain = Domain([-10, 10], continuous=True)

rv = RV(domain)

data = {rv: np.random.normal(loc=0, scale=1, size=100)}

nn = NeuralNetPotential(
    (1, 16, LeakyReLU()),
    (16, 16, LeakyReLU()),
    (16, 1, None)
)

f = F(nn, nb=[rv])

g = Graph({rv}, {f})

visualize_1d_neural_net(nn, domain, 1)

leaner = PseudoMLELearner(g, {nn}, data)
leaner.train(
    lr=0.0001,
    regular=1,
    max_iter=100,
    batch_iter=10,
    batch_size=100,
    rvs_selection_size=1,
    sample_size=10
)

visualize_1d_neural_net(nn, domain, 1)
