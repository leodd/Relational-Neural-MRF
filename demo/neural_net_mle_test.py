from NeuralNetPotential import NeuralNetPotential, LeakyReLU, ReLU, ELU
from learning.PseudoMLE import PseudoMLELearner
from utils import visualize_1d_neural_net
from Graph import *
import numpy as np
import seaborn as sns


domain = Domain([-20, 10], continuous=True)

rv = RV(domain)

data = {
    rv: np.concatenate([
        np.random.normal(loc=0, scale=3, size=300),
        np.random.normal(loc=-10, scale=3, size=300)
    ])
}

sns.distplot(data[rv])

nn = NeuralNetPotential(
    (1, 64, ReLU()),
    (64, 64, ReLU()),
    (64, 64, ReLU()),
    (64, 1, None)
)

f = F(nn, nb=[rv])

g = Graph({rv}, {f})

visualize_1d_neural_net(nn, domain, 0.3, True)
visualize_1d_neural_net(nn, domain, 0.3, False)

leaner = PseudoMLELearner(g, {nn}, data)
leaner.train(
    lr=0.0005,
    alpha=0.5,
    regular=0.001,
    max_iter=5000,
    batch_iter=10,
    batch_size=300,
    rvs_selection_size=1,
    sample_size=10
)

visualize_1d_neural_net(nn, domain, 0.3, True)
visualize_1d_neural_net(nn, domain, 0.3, False)
