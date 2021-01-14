from functions.NeuralNetPotentialTorchVersion import NeuralNetPotential
from learner.torch.PseudoMLETorchVersion import PseudoMLELearner
from utils import visualize_1d_potential_torch
from Graph import *
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns


device = torch.device('cuda:0')

domain = Domain([-20, 10], continuous=True)

rv = RV(domain)

data = {
    rv: np.concatenate([
        np.random.normal(loc=0, scale=1, size=1000),
        np.random.normal(loc=-10, scale=1, size=1000)
    ])
}

sns.distplot(data[rv])

p = NeuralNetPotential(
    (1, 64, nn.ReLU()),
    (64, 32, nn.ReLU()),
    (32, 1, None),
    device=device
)

f = F(p, nb=[rv])

g = Graph({rv}, {f})

visualize_1d_potential_torch(p.nn, domain, 0.3, True)
visualize_1d_potential_torch(p.nn, domain, 0.3, False)

leaner = PseudoMLELearner(g, {p}, data, device=device)
leaner.train(
    lr=0.01,
    alpha=0.5,
    regular=0.001,
    max_iter=5000,
    batch_iter=10,
    batch_size=300,
    rvs_selection_size=1,
    sample_size=20
)

visualize_1d_potential_torch(p.nn, domain, 0.3, True)
visualize_1d_potential_torch(p.nn, domain, 0.3, False)
