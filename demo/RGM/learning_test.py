from demo.RGM.rgm_generator import *
from NeuralNetPotential import GaussianNeuralNetPotential, ReLU
from learning.PseudoMLEWithPrior import PseudoMLELearner
from utils import visualize_2d_potential


p1 = GaussianNeuralNetPotential(
    (2, 64, ReLU()),
    (64, 32, ReLU()),
    (32, 1, None)
)

p2 = GaussianNeuralNetPotential(
    (2, 64, ReLU()),
    (64, 32, ReLU()),
    (32, 1, None)
)

p3 = GaussianNeuralNetPotential(
    (2, 64, ReLU()),
    (64, 32, ReLU()),
    (32, 1, None)
)

rel_g = generate_rel_graph(p1, p2, p3)
g, rvs_dict = rel_g.ground_graph()

# conditional_rvs = {rvs_dict[('recession', 'all')]}
# g.condition_rvs = conditional_rvs

data = dict()
for key, value in load_data('rgm-joint').items():
    data[rvs_dict[key]] = value

leaner = PseudoMLELearner(g, {p1, p2, p3}, data)
leaner.train(
    lr=0.01,
    alpha=0.8,
    regular=0.0001,
    max_iter=10000,
    batch_iter=20,
    batch_size=300,
    rvs_selection_size=100,
    sample_size=20
)

domain = Domain([-10, 10], continuous=True)
visualize_2d_potential(p1, domain, domain, 2)
visualize_2d_potential(p2, domain, domain, 2)
visualize_2d_potential(p3, domain, domain, 2)
